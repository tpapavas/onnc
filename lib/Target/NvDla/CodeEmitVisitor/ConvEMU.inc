PP_DEFINE_VISIT_CPU(Conv, pOp)
{
  const Tensor* input_X_t       = pOp.getInput(0);
  int32_t       input_X_ndim    = input_X_t->getNumOfDimensions();
  int32_t       input_X_dims[4] = {1, 1, 1, 1};
  for (int i = 0; i < input_X_ndim; ++i)
    input_X_dims[i] = input_X_t->dimension(i);

  const Tensor* input_W_t       = pOp.getInput(1);
  int32_t       input_W_ndim    = input_W_t->getNumOfDimensions();
  int32_t       input_W_dims[4] = {1, 1, 1, 1};
  for (int i = 0; i < input_W_ndim; ++i)
    input_W_dims[i] = input_W_t->dimension(i);

  const Tensor* const     input_B_t    = (pOp.hasBias() ? pOp.getB() : nullptr);
  const Tensor::Dimension input_B_size = (pOp.hasBias() ? input_B_t->dimension(0) : 1);

  const Tensor* output_Y_t       = pOp.getOutput(0);
  int32_t       output_Y_ndim    = output_Y_t->getNumOfDimensions();
  int32_t       output_Y_dims[4] = {1, 1, 1, 1};
  for (int i = 0; i < output_Y_ndim; ++i)
    output_Y_dims[i] = output_Y_t->dimension(i);
  NvDlaCubeInfo Y_cube(*this, NVDLA_CUBE_FEATURE, output_Y_dims[0], output_Y_dims[1], output_Y_dims[2],
                       output_Y_dims[3]);

  // INT8 requantization. Build the per-output-channel rescale
  //   R[k] = SF_in * SF_wt[k] / SF_out
  // and decompose it into NVDLA's (shared shift, per-channel int16 multiplier)
  // form. The multipliers are applied as the SDP per-kernel MUL operand
  // (packRequantMul) and the shift as the SDP X1 truncate. This mirrors
  // nvdla_compiler exactly (verified against nvdla/sw, see docs/ONNC_INT8_PLAN.md
  // Part D): the conv does NO pre-shift -- it hands the int32 accumulator to the
  // SDP (ConvCoreNode.cpp: setOutTruncate(0)/setPraTruncate(0)) -- and the whole
  // shared shift lives in the SDP X1 stage (ScaleOp.cpp: x1Params().setTruncate).
  bool                      int8Requant = false;
  std::vector<std::int16_t> requantMult;     // per output channel, m[k] (int16) -> SDP X1 MUL
  std::vector<std::int16_t> requantBias;     // per output channel, round(b[k]/SF_out) (int16) -> SDP X2 ADD
  int                       sdpShift    = 0;  // SDP x1_op.truncate (full shared shift, UINT5)
  if (DLA_PRECISION == PRECISION_INT8) {
    const float sfIn  = getActivationScale(input_X_t->getName());
    // Resolve across a trailing scale-preserving reshape so the gemm->conv final
    // layer (output `replace_gemm_by_conv_N`, reshaped to the calibrated graph
    // output) gets its real SF_out instead of 0 -> identity -> int8 saturation.
    const float sfOut = getActivationScaleResolved(output_Y_t);

    const std::vector<float>* wtScales = getWeightScales(input_W_t);
    if (wtScales == nullptr) {
      if (const FloatTensor* const wft = dynamic_cast<const FloatTensor*>(input_W_t)) {
        computeWeightScales(input_W_t, wft->getValues(), NvDlaDims(*input_W_t));
        wtScales = getWeightScales(input_W_t);
      }
    }

    if (sfIn > 0.0f && sfOut > 0.0f && wtScales != nullptr && !wtScales->empty()) {
      std::vector<double> ratios(wtScales->size());
      for (std::size_t k = 0; k < wtScales->size(); ++k) {
        ratios[k] = static_cast<double>(sfIn) * (*wtScales)[k] / static_cast<double>(sfOut);
      }
      const internal::PerChannelCvt pc = internal::realToPerChannelCvt(ratios);

      int8Requant = true;
      requantMult = pc.mult;
      sdpShift    = static_cast<int>(pc.truncate); // full shift -> SDP X1 (no conv pre-shift)

      // Per-channel bias in the OUTPUT (SF_out) domain, added in SDP X2 after the
      // X1 rescale: bias_out[k] = round(b_fp32[k] / SF_out), int16. Matches
      // nvdla_compiler's fused Scale(X1)+Bias(X2) (BatchNormOp.cpp): the bias adds
      // in the post-rescale (output) domain. See docs/ONNC_INT8_PLAN.md B.6.
      if (pOp.hasBias()) {
        if (const FloatTensor* const bft = dynamic_cast<const FloatTensor*>(input_B_t)) {
          const auto biasVals = bft->getValues();
          requantBias.resize(biasVals.size());
          for (std::size_t k = 0; k < biasVals.size(); ++k) {
            double q = std::round(static_cast<double>(biasVals[k]) / static_cast<double>(sfOut));
            q       = std::max(-32768.0, std::min(32767.0, q));
            requantBias[k] = static_cast<std::int16_t>(q);
          }
        }
      }

      outs() << "[NvDla][int8] requant " << output_Y_t->getName() << " : SF_in=" << sfIn << " SF_out=" << sfOut
             << " ch=" << ratios.size() << " x1_truncate=" << sdpShift
             << " bias=" << (requantBias.empty() ? "no" : "int16/out-domain") << "\n";
    } else {
      outs() << "[NvDla][int8] requant " << output_Y_t->getName() << " : missing scales -> identity (SF_in=" << sfIn
             << " SF_out=" << sfOut << " have_wt=" << (wtScales != nullptr) << ")\n";
    }
  }

  // Prepare attributes
  int32_t number_of_dilations = pOp.getDilations().vector().size();
  int32_t dilations[2]        = {1, 1};
  for (int i = 0; i < number_of_dilations; ++i)
    dilations[i] = pOp.getDilations().at(i);
  const int32_t numGroups              = pOp.getGroup().value();
  int32_t       number_of_kernel_shape = pOp.getKernelShape().vector().size();
  int32_t       kernel_shape[2]        = {1, 1};
  for (int i = 0; i < number_of_kernel_shape; ++i)
    kernel_shape[i] = pOp.getKernelShape().at(i);
  int32_t number_of_pads = pOp.getPads().vector().size();
  int32_t pads[4]        = {0, 0, 0, 0};
  for (int i = 0; i < number_of_pads; ++i)
    pads[i] = pOp.getPads().at(i);
  int32_t number_of_strides = pOp.getStrides().vector().size();
  int32_t strides[2]        = {1, 1};
  for (int i = 0; i < number_of_strides; ++i)
    strides[i] = pOp.getStrides().at(i);

  NvDlaCubeInfo X_cube(*this, NVDLA_CUBE_FEATURE, input_X_dims[0],
                       input_X_dims[1], input_X_dims[2], input_X_dims[3], pads[1], pads[3]);

  input_X_dims[1] /= numGroups;
  input_W_dims[0] /= numGroups;

  NvDlaCubeInfo fcube_group(*this, NVDLA_CUBE_FEATURE, input_X_dims[0],
                            input_X_dims[1], input_X_dims[2], input_X_dims[3], pads[1], pads[3]);
  const Tensor::Dimension numInputChannelsPerGroup  = input_X_dims[1];
  const Tensor::Dimension numOutputChannelsPerGroup = input_W_dims[0];
  for (std::int32_t groupIdx = 0; groupIdx < numGroups; ++groupIdx) {
    // output/input channel offsets w/o alignment by atomic k
    const Tensor::Dimension inputChannelOffset        = groupIdx * numInputChannelsPerGroup;
    const Tensor::Dimension alignedInputChannelOffset = ((inputChannelOffset / MAC_ATOMIC_K) * MAC_ATOMIC_K);
    const Tensor::Dimension outputChannelOffset       = groupIdx * numOutputChannelsPerGroup;
    const Tensor::Dimension alignedOutputChannelOffset =
      groupIdx * UNIT_ALIGNMENT(numOutputChannelsPerGroup, MAC_ATOMIC_K);

    const Tensor::Dimension numWeightFrontPaddingChannels = inputChannelOffset - alignedInputChannelOffset;

    NvDlaCubeInfo winfo(*this, NVDLA_CUBE_WEIGHT, input_W_dims[0],
                        numWeightFrontPaddingChannels + numInputChannelsPerGroup, input_W_dims[2], input_W_dims[3]);

    // Weight Memory allocation, repacking by groups
    int W_mid = -1;
    W_mid = packWeight(*input_W_t, NvDlaDims(input_W_dims), numWeightFrontPaddingChannels, outputChannelOffset);

    // int W_addr = issueDlaAddr(W_mid, winfo);
    int W_addr = issueEmuAddr(W_mid);
    int B_mid  = -1;
    int B_addr = -1;

    NvDlaCubeInfo B_info(*this, NVDLA_CUBE_FEATURE, numOutputChannelsPerGroup, 1, 1, 1);

    // FP16 / non-requant path uses the fp16 bias here; INT8 requant uses the
    // int16 output-domain bias (Bq) packed below, so skip this zero blob then.
    if (pOp.hasBias() && !int8Requant) {
      B_mid  = packBias(*input_B_t, numOutputChannelsPerGroup, outputChannelOffset);
      // B_addr = issueDlaAddr(B_mid, B_info);
      B_addr = issueEmuAddr(B_mid);
    }

    // INT8: emit this group's per-kernel requant multipliers as a 2-byte SDP
    // operand cube (consumed by the bias-add SDP's x1 MUL stage below).
    int           M_mid  = -1;
    int           M_addr = -1;
    NvDlaCubeInfo M_info(*this, NVDLA_CUBE_SDP_X_ALU_OR_MUL_TWO_BYTE, 1, numOutputChannelsPerGroup, 1, 1);
    if (int8Requant) {
      M_mid  = packRequantMul(requantMult, numOutputChannelsPerGroup, outputChannelOffset);
      // M_addr = issueDlaAddr(M_mid, M_info);
      M_addr = issueEmuAddr(M_mid);
    }

    // INT8: this group's per-kernel int16 bias (output domain) for the SDP X2 ADD.
    int Bq_mid  = -1;
    int Bq_addr = -1;
    if (int8Requant && pOp.hasBias() && !requantBias.empty()) {
      Bq_mid  = packRequantMul(requantBias, numOutputChannelsPerGroup, outputChannelOffset);
      // Bq_addr = issueDlaAddr(Bq_mid, M_info);
      Bq_addr = issueEmuAddr(Bq_mid);
    }

    int pad_top       = pads[0];
    int pad_left      = pads[1];
    int pad_bottom    = pads[2];
    int pad_right     = pads[3];
    int kernel_height = input_W_dims[2] + (dilations[0] - 1) * (input_W_dims[2] - 1);
    int stride_y      = strides[0];
    int input_height  = input_X_dims[2];
    int output_height = output_Y_dims[2];
    assert(output_height == ((input_height + pad_top + pad_bottom - kernel_height) / stride_y + 1));

    unsigned min_data_banks_needed = 0;
    bool     is_weight_reuse       = false;
    std::tie(min_data_banks_needed, is_weight_reuse) =
      tryAllocateDataAndWeightsIntoCBuf(fcube_group, winfo, dilations[0]);

    int affordable_conv_height = (CBUF_BANK_DEPTH * (CBUF_BANK_NUM - winfo.banks)) / fcube_group.eps;
    // int affordable_conv_height = 256;

    int data_banks_of_first_split_layer = -1;
    int input_h_idx         = -pad_top; // the starting H where the input data of each split convolution comes from.
    int output_h_idx        = 0;        // the starting H where the output data of each split convolution goes to.
    int is_first_split      = true;
    int is_last_split       = false;
    int unused_input_height = (input_height + pad_top) - (kernel_height + stride_y * (output_height - 1));
    if (unused_input_height < 0)
      unused_input_height = 0;

    do { // for each split convolution
      is_last_split = (input_height - unused_input_height - std::max(input_h_idx, 0)) <= affordable_conv_height;

      int input_split_height;
      int output_split_height;
      if (is_first_split && is_last_split) { // No split
        output_split_height = output_height;
        input_split_height  = input_height;
      } else if (is_last_split) {
        input_split_height  = input_height - unused_input_height - std::max(input_h_idx, 0);
        output_split_height = output_height - output_h_idx;
        assert(output_split_height == ((input_split_height + pad_bottom - kernel_height) / stride_y + 1));
      } else if (is_first_split) {
        output_split_height = (affordable_conv_height + pad_top - kernel_height) / stride_y + 1;
        input_split_height  = kernel_height + stride_y * (output_split_height - 1) - pad_top;
      } else { // Split convolution in middle
        output_split_height = (affordable_conv_height - kernel_height) / stride_y + 1;
        input_split_height  = kernel_height + stride_y * (output_split_height - 1);
      }

      int split_pad_top    = (is_first_split) ? pad_top : 0;
      int split_pad_bottom = (is_last_split) ? pad_bottom : 0;

      NvDlaCubeInfo finfo(*this, NVDLA_CUBE_FEATURE, input_X_dims[0],
                          numWeightFrontPaddingChannels + numInputChannelsPerGroup, input_split_height,
                          input_X_dims[3], pads[1], pads[3]);

      if (is_first_split) {
        data_banks_of_first_split_layer = finfo.banks;
      } else {                 // The rest split layers
        if (is_weight_reuse) { // All weights are buffered in the CBUF, so weights are reused among split layers.
          if (!(finfo.banks <= data_banks_of_first_split_layer)) {
            fatal(nvdla_unexpected_num_of_banks) << PP_STRINGIFY(finfo.banks) << finfo.banks;
          }
          // All split layers must have the same data_bank setting.
          finfo.banks = data_banks_of_first_split_layer;
        }
      }

      NvDlaCubeInfo oinfo(*this, NVDLA_CUBE_FEATURE, output_Y_dims[0], numOutputChannelsPerGroup, output_split_height,
                          output_Y_dims[3]);

      // NvDlaDlaOperation* const conv_op = new NvDlaDlaOperation();
      NvDlaEmuOperation* const conv_op = new NvDlaEmuOperation();
      // conv_op->op_dep.op_type          = DLA_OP_CONV;

      // struct dla_conv_op_desc* conv_desc = (struct dla_conv_op_desc*)(&(conv_op->op_desc));
      struct emu_conv_op_desc* conv_desc = (struct emu_conv_op_desc*)(&(conv_op->op_desc));
      struct emu_common_op_desc          common;
      std::memset(&common, 0, sizeof(common));
      common.op_type             = NVDLA_EMU_OP_CONV;
      conv_desc->common = common;

      conv_desc->conv_mode               = CONV_MODE_DIRECT;
      conv_desc->data_reuse              = 0;
      conv_desc->weight_reuse            = (is_weight_reuse && !is_first_split) ? 1 : 0;
      conv_desc->skip_data_rls           = 0;
      conv_desc->skip_weight_rls         = (is_weight_reuse && !is_last_split) ? 1 : 0;
      conv_desc->entry_per_slice         = finfo.eps;
      conv_desc->data_format = FORMAT_FEATURE;
      conv_desc->pixel_mapping = 0;
      conv_desc->fetch_grain   = 1;
      conv_desc->batch         = 1;
      conv_desc->weight_format = WEIGHT_FORMAT_UNCOMPRESSED;
      conv_desc->data_bank     = finfo.banks;
      conv_desc->weight_bank   = winfo.banks;
      conv_desc->batch_stride   = 0;
      conv_desc->post_extension = 0;
      conv_desc->pixel_override = 0;
      conv_desc->release = input_split_height;
      conv_desc->input_width_csc    = finfo.dim_w;
      conv_desc->input_height_csc   = finfo.dim_h;
      conv_desc->input_channel_csc  = finfo.dim_c;
      conv_desc->kernel_channel_csc = winfo.dim_c;
      conv_desc->kernel_width_csc   = winfo.dim_w;
      conv_desc->kernel_height_csc  = winfo.dim_h;
      conv_desc->input_width_cmac   = output_Y_dims[3];
      conv_desc->input_height_cmac  = output_split_height;
      conv_desc->bytes_per_kernel   = winfo.dim_c * winfo.dim_h * winfo.dim_w * ELEMENT_SIZE;
      conv_desc->mean_ry            = 0;
      conv_desc->mean_gu            = 0;
      conv_desc->mean_bv            = 0;
      conv_desc->mean_ax            = 0;
      conv_desc->mean_format        = 0;
      conv_desc->conv_stride_x      = strides[1];
      conv_desc->conv_stride_y      = strides[0];
      conv_desc->pad_x_left         = pad_left;
      conv_desc->pad_x_right        = pad_right;
      conv_desc->pad_y_top          = split_pad_top;
      conv_desc->pad_y_bottom       = split_pad_bottom;
      conv_desc->dilation_x         = dilations[1];
      conv_desc->dilation_y         = dilations[0];
      conv_desc->pra_truncate       = 0;
      // conv_desc->in_precision       = DLA_PRECISION;
      /**
       * Temporary solution.
       * Attempt to map dla_interface precision consts
       * to emu_interface ones.
       */
      conv_desc->in_precision       = DLA_PRECISION + 1;
      conv_desc->out_precision      = DLA_PRECISION;
      conv_desc->out_cvt_scale      = 1;
      // No conv pre-shift: the conv hands the int32 accumulator to the SDP and the
      // whole rescale (per-channel MUL + shift) happens there -- matching
      // nvdla_compiler (ConvCoreNode.cpp sets out/pra truncate to 0).
      conv_desc->out_cvt_truncate   = 0;
      conv_desc->out_cvt_enable     = 1;
      conv_desc->pad_val            = 0;

      conv_desc->has_relu           = pOp.hasRelu() ? 1 : 0;
      // conv_desc->relu_op_cvt_scale  = pOp.hasRelu() && int8Requant  ? (int16_t) pOp.getReluScale().value() : 0;
      // conv_desc->relu_op_cvt_truncate = pOp.hasRelu() && int8Requant ? (int16_t) pOp.getReluTruncate().value() : 0;

      NvU16 emuFormat = EMU_FORMAT_FF16;
      if (DLA_PRECISION == PRECISION_INT8)
        emuFormat = (FEATURE_ATOM_CUBE_SIZE == 8) ? EMU_FORMAT_INT8_8 : EMU_FORMAT_INT8;

      struct emu_conv_buffer_descs* conv_surf = (struct emu_conv_buffer_descs*)(&(conv_op->op_buf));
      // conv_surf->weight_data.type             = DLA_MEM_MC;
      // conv_surf->weight_data.address          = W_addr;
      conv_surf->weight_data.addressIndex     = W_addr;
      conv_surf->weight_data.size             = m_pMeta.getMemoryListEntrySize(W_mid);
      conv_surf->weight_data.format           = emuFormat; // 3: fp16
      conv_surf->weight_data.width            = winfo.dim_w;
      conv_surf->weight_data.height           = winfo.dim_h;
      conv_surf->weight_data.channel          = winfo.dim_c;
      conv_surf->weight_data.line_stride      = 0;
      conv_surf->weight_data.surf_stride      = 0;
      // conv_surf->weight_data.plane_stride     = 0;

      // conv_surf->wmb_data.type    = DLA_MEM_HW;
      // conv_surf->wmb_data.address = -1;
      conv_surf->wmb_data.addressIndex = -1;

      // conv_surf->wgs_data.type    = DLA_MEM_HW;
      // conv_surf->wgs_data.address = -1;
      conv_surf->wgs_data.addressIndex = -1;

      // conv_surf->src_data.type = DLA_MEM_MC;
      // conv_surf->src_data.address =
      //   issueDlaAddr(*input_X_t, X_cube, alignedInputChannelOffset, std::max(input_h_idx, 0));
      int input_mid = m_pMeta.getMemoryListEntryId(*input_X_t);
      conv_surf->src_data.addressIndex =
        issueEmuAddr(input_mid);
      conv_surf->src_data.size         = finfo.size;
      conv_surf->src_data.format       = 3; // fp16
      conv_surf->src_data.width        = finfo.dim_w;
      conv_surf->src_data.height       = finfo.dim_h;
      conv_surf->src_data.channel      = finfo.dim_c;
      printf("CONV src dimentions: (w, h, c) %d, %d, %d'\n", conv_surf->src_data.width, conv_surf->src_data.height, conv_surf->src_data.channel);
      conv_surf->src_data.line_stride  = fcube_group.stride_line;
      conv_surf->src_data.surf_stride  = fcube_group.stride_surface;
      // conv_surf->src_data.plane_stride = fcube_group.stride_plane;

      // conv_surf->dst_data.type         = DLA_MEM_HW;
      // conv_surf->dst_data.address      = -1;
      conv_surf->dst_data.addressIndex = -1;
      conv_surf->dst_data.format       = 3; // fp16
      conv_surf->dst_data.size         = oinfo.size;
      conv_surf->dst_data.width        = oinfo.dim_w;
      conv_surf->dst_data.height       = oinfo.dim_h;
      conv_surf->dst_data.channel      = oinfo.dim_c;
      conv_surf->dst_data.line_stride  = Y_cube.stride_line;
      conv_surf->dst_data.surf_stride  = Y_cube.stride_surface;
      // conv_surf->dst_data.plane_stride = Y_cube.stride_plane;

      // Bias Add
      /*
      // NvDlaDlaOperation* const add_op = new NvDlaDlaOperation();
      // add_op->op_dep.op_type          = DLA_OP_SDP;

      struct dla_sdp_op_desc* add_desc = (struct dla_sdp_op_desc*)(&(add_op->op_desc));
      add_desc->src_precision          = DLA_PRECISION;
      add_desc->dst_precision          = DLA_PRECISION;
      add_desc->lut_index              = -1;
      add_desc->conv_mode              = 0;
      add_desc->out_cvt.scale          = 1;
      // INT8: the SDP out_cvt stays identity; the residual shift S lives in the X1
      // stage (x1_op.truncate) so it is applied right after the per-kernel MUL and
      // BEFORE the X2 bias add (which is therefore in the SF_out domain) -- this is
      // how nvdla_compiler programs the per-kernel scale (x1Params().setTruncate).
      add_desc->out_cvt.truncate       = 0;
      add_desc->out_cvt.enable         = 1;
      add_desc->out_cvt.offset         = 0;
      add_desc->conv_mode              = CONV_MODE_DIRECT;
      add_desc->batch_num              = 1;
      add_desc->batch_stride           = 0;
      if (int8Requant) {
        // x1 = per-kernel MUL: apply the per-channel requant multipliers m[k]
        // (int16 operand cube). x2 = per-kernel ALU SUM: the bias (zero until the
        // int8-bias patch). Order conv -> MUL(rescale) -> ADD(bias) -> out_cvt.
        add_desc->x1_op.enable               = 1;
        add_desc->x1_op.alu_type             = SDP_ALU_OP_SUM;
        add_desc->x1_op.type                 = SDP_OP_MUL;
        add_desc->x1_op.mode                 = SDP_OP_PER_KERNEL;
        add_desc->x1_op.act                  = ACTIVATION_NONE;
        add_desc->x1_op.shift_value          = 0;
        add_desc->x1_op.truncate             = static_cast<std::uint8_t>(sdpShift); // full shared shift (UINT5)
        add_desc->x1_op.precision            = PRECISION_INT16; // 2-byte per-kernel operand
        add_desc->x1_op.alu_operand          = 0;
        add_desc->x1_op.mul_operand          = 1;
        add_desc->x1_op.cvt.alu_cvt.scale    = 0;
        add_desc->x1_op.cvt.alu_cvt.truncate = 0;
        add_desc->x1_op.cvt.alu_cvt.enable   = 0;
        add_desc->x1_op.cvt.alu_cvt.offset   = 0;
        add_desc->x1_op.cvt.mul_cvt.scale    = 1;
        add_desc->x1_op.cvt.mul_cvt.truncate = 0;
        add_desc->x1_op.cvt.mul_cvt.enable   = 0;
        add_desc->x1_op.cvt.mul_cvt.offset   = 0;

        if (pOp.hasBias()) {
          add_desc->x2_op.enable               = 1;
          add_desc->x2_op.alu_type             = SDP_ALU_OP_SUM;
          add_desc->x2_op.type                 = SDP_OP_ALU;
          add_desc->x2_op.mode                 = SDP_OP_PER_KERNEL;
          add_desc->x2_op.act                  = ACTIVATION_NONE;
          add_desc->x2_op.shift_value          = 0;
          add_desc->x2_op.truncate             = 0;
          add_desc->x2_op.precision            = PRECISION_INT16; // int16 per-kernel bias operand
          add_desc->x2_op.alu_operand          = 0;
          add_desc->x2_op.mul_operand          = 0;
          add_desc->x2_op.cvt.alu_cvt.scale    = 0;
          add_desc->x2_op.cvt.alu_cvt.truncate = 0;
          add_desc->x2_op.cvt.alu_cvt.enable   = 0;
          add_desc->x2_op.cvt.alu_cvt.offset   = 0;
          add_desc->x2_op.cvt.mul_cvt.scale    = 0;
          add_desc->x2_op.cvt.mul_cvt.truncate = 0;
          add_desc->x2_op.cvt.mul_cvt.enable   = 0;
          add_desc->x2_op.cvt.mul_cvt.offset   = 0;
        } else {
          add_desc->x2_op.enable = 0;
        }
        add_desc->y_op.enable = 0;
      } else if (pOp.hasBias()) {
        add_desc->x1_op.enable               = 1;
        add_desc->x1_op.alu_type             = SDP_ALU_OP_SUM;
        add_desc->x1_op.type                 = SDP_OP_ALU;
        add_desc->x1_op.mode                 = SDP_OP_PER_KERNEL;
        add_desc->x1_op.act                  = ACTIVATION_NONE;
        add_desc->x1_op.shift_value          = 0;
        add_desc->x1_op.truncate             = 0;
        add_desc->x1_op.precision            = DLA_PRECISION;
        add_desc->x1_op.alu_operand          = 0;
        add_desc->x1_op.mul_operand          = 0;
        add_desc->x1_op.cvt.alu_cvt.scale    = 0;
        add_desc->x1_op.cvt.alu_cvt.truncate = 0;
        add_desc->x1_op.cvt.alu_cvt.enable   = 0;
        add_desc->x1_op.cvt.alu_cvt.offset   = 0;
        add_desc->x1_op.cvt.mul_cvt.scale    = 0;
        add_desc->x1_op.cvt.mul_cvt.truncate = 0;
        add_desc->x1_op.cvt.mul_cvt.enable   = 0;
        add_desc->x1_op.cvt.mul_cvt.offset   = 0;
        add_desc->x2_op.enable = 0;
        add_desc->y_op.enable  = 0;
      } else {
        add_desc->x1_op.enable = 0;
        add_desc->x2_op.enable = 0;
        add_desc->y_op.enable  = 0;
      }


      */

      conv_desc->src_precision          = DLA_PRECISION;
      conv_desc->dst_precision          = DLA_PRECISION;
      conv_desc->lut_index              = -1;
      conv_desc->conv_mode              = 0;
      // conv_desc->out_cvt_scale          = 1;
      // conv_desc->out_cvt_truncate       = 0;
      // conv_desc->out_cvt_enable         = 1;
      // conv_desc->out_cvt_offset         = 0;
      conv_desc->conv_mode              = CONV_MODE_DIRECT;
      conv_desc->batch_num              = 1;
      // conv_desc->batch_stride           = 0;
      if (pOp.hasBias()) {
        conv_desc->x1_op_enable               = 1;
        conv_desc->x1_op_alu_type             = SDP_ALU_OP_SUM;
        conv_desc->x1_op_type                 = SDP_OP_ALU;
        conv_desc->x1_op_mode                 = SDP_OP_PER_KERNEL;
        conv_desc->x1_op_act                  = ACTIVATION_NONE;
        conv_desc->x1_op_shift_value          = 0;
        conv_desc->x1_op_truncate             = 0;
        conv_desc->x1_op_precision            = DLA_PRECISION;
        conv_desc->x1_op_alu_operand          = 0;
        conv_desc->x1_op_mul_operand          = 0;
        conv_desc->x1_op_cvt_alu_cvt_scale    = 0;
        conv_desc->x1_op_cvt_alu_cvt_truncate = 0;
        conv_desc->x1_op_cvt_alu_cvt_enable   = 0;
        conv_desc->x1_op_cvt_alu_cvt_offset   = 0;
        conv_desc->x1_op_cvt_mul_cvt_scale    = 0;
        conv_desc->x1_op_cvt_mul_cvt_truncate = 0;
        conv_desc->x1_op_cvt_mul_cvt_enable   = 0;
        conv_desc->x1_op_cvt_mul_cvt_offset   = 0;
      } else {
        conv_desc->x1_op_enable = 0;
      }

      // struct dla_sdp_surface_desc* add_surf = (struct dla_sdp_surface_desc*)(&(add_op->op_surf));
      // add_surf->src_data.type               = DLA_MEM_HW;
      // add_surf->src_data.address            = -1;
      // add_surf->src_data.size               = conv_surf->dst_data.size;
      // add_surf->src_data.width              = conv_surf->dst_data.width;
      // add_surf->src_data.height             = conv_surf->dst_data.height;
      // add_surf->src_data.channel            = conv_surf->dst_data.channel;
      // add_surf->src_data.line_stride        = 0;
      // add_surf->src_data.surf_stride        = 0;
      // add_surf->src_data.plane_stride       = 0;

      if (int8Requant) {
        // x1 <- per-kernel requant multipliers (int16 cube); x2 <- bias.
        // add_surf->x1_data.type         = DLA_MEM_MC;
        // add_surf->x1_data.address      = M_addr;
        // add_surf->x1_data.size         = m_pMeta.getMemoryListEntrySize(M_mid);
        // add_surf->x1_data.width        = 1;
        // add_surf->x1_data.height       = 1;
        // add_surf->x1_data.channel      = numOutputChannelsPerGroup;
        // add_surf->x1_data.line_stride  = M_info.stride_line;
        // add_surf->x1_data.surf_stride  = M_info.stride_surface;
        // add_surf->x1_data.plane_stride = M_info.stride_plane;

        /**
         * NOTE: Hacky solution.
         * x1_op_truncate is originally for bias (x1) scale.
         * As we don't use x1 in ConvEMU we can utilize it
         * for conv requantization.
         */
        conv_desc->x1_op_truncate            = static_cast<std::uint8_t>(sdpShift); // full shared shift (UINT5)

        conv_surf->requant_data.addressIndex = M_addr;
        conv_surf->requant_data.size         = m_pMeta.getMemoryListEntrySize(M_mid);
        conv_surf->requant_data.format       = 3; // fp16
        conv_surf->requant_data.width        = 1;
        conv_surf->requant_data.height       = 1;
        conv_surf->requant_data.channel      = numOutputChannelsPerGroup;
        conv_surf->requant_data.line_stride  = M_info.stride_line;
        conv_surf->requant_data.surf_stride  = M_info.stride_surface;
        conv_surf->requant_data.plane_stride = M_info.stride_plane;
        if (pOp.hasBias()) {
          // add_surf->x2_data.type         = DLA_MEM_MC;
          // add_surf->x2_data.address      = Bq_addr; // int16 output-domain bias cube
          // add_surf->x2_data.size         = m_pMeta.getMemoryListEntrySize(Bq_mid);
          // add_surf->x2_data.width        = 1;
          // add_surf->x2_data.height       = 1;
          // add_surf->x2_data.channel      = numOutputChannelsPerGroup;
          // add_surf->x2_data.line_stride  = M_info.stride_line;
          // add_surf->x2_data.surf_stride  = M_info.stride_surface;
          // add_surf->x2_data.plane_stride = M_info.stride_plane;

          conv_surf->bias_data.addressIndex = Bq_addr; // int16 output-domain bias cube
          conv_surf->bias_data.size         = m_pMeta.getMemoryListEntrySize(Bq_mid);
          conv_surf->bias_data.format       = 3; // fp16
          conv_surf->bias_data.width        = 1;
          conv_surf->bias_data.height       = 1;
          conv_surf->bias_data.channel      = numOutputChannelsPerGroup;
          conv_surf->bias_data.line_stride  = M_info.stride_line;
          conv_surf->bias_data.surf_stride  = M_info.stride_surface;
          // conv_surf->bias_data.plane_stride = M_info.stride_plane;
        }
      } else if (pOp.hasBias()) {
      // if (pOp.hasBias()) {
        // add_surf->x1_data.type         = DLA_MEM_MC;
        // add_surf->x1_data.address      = B_addr;
        // add_surf->x1_data.size         = m_pMeta.getMemoryListEntrySize(B_mid);
        // add_surf->x1_data.width        = 1;
        // add_surf->x1_data.height       = 1;
        // add_surf->x1_data.channel      = numOutputChannelsPerGroup;
        // add_surf->x1_data.line_stride  = B_info.stride_line;
        // add_surf->x1_data.surf_stride  = B_info.stride_surface;
        // add_surf->x1_data.plane_stride = B_info.stride_plane;

        
        conv_surf->bias_data.addressIndex = B_addr;
        conv_surf->bias_data.size         = m_pMeta.getMemoryListEntrySize(B_mid);
        conv_surf->bias_data.format       = 3; // fp16
        conv_surf->bias_data.width        = 1;
        conv_surf->bias_data.height       = 1;
        conv_surf->bias_data.channel      = numOutputChannelsPerGroup;
        conv_surf->bias_data.line_stride  = B_info.stride_line;
        conv_surf->bias_data.surf_stride  = B_info.stride_surface;
      }

      // add_surf->dst_data.type         = DLA_MEM_MC;
      // add_surf->dst_data.address      = issueDlaAddr(*output_Y_t, Y_cube, alignedOutputChannelOffset, output_h_idx);
      int output_mid = m_pMeta.getMemoryListEntryId(*output_Y_t);
      conv_surf->dst_data.addressIndex   = issueEmuAddr(output_mid);
      // add_surf->dst_data.size         = conv_surf->dst_data.size;
      // add_surf->dst_data.width        = conv_surf->dst_data.width;
      // add_surf->dst_data.height       = conv_surf->dst_data.height;
      // add_surf->dst_data.channel      = conv_surf->dst_data.channel;
      // add_surf->dst_data.line_stride  = conv_surf->dst_data.line_stride;
      // add_surf->dst_data.surf_stride  = conv_surf->dst_data.surf_stride;
      // add_surf->dst_data.plane_stride = conv_surf->dst_data.plane_stride;

      // NvDlaDlaOperation* prev_op = (is_first_split) ? m_pMeta.m_pPrevOp : NULL;
      // issueDlaOp(conv_op, add_op, prev_op);
      issueEmuOp(conv_op);

      output_h_idx = output_h_idx + output_split_height;
      input_h_idx  = stride_y * output_h_idx - pad_top;

      is_first_split = false;
    } while (!is_last_split);
  }
}
