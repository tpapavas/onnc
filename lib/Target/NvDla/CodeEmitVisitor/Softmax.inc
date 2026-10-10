PP_DEFINE_VISIT(Softmax, pOp)
{
  using namespace internal;

  m_pMeta.m_DlaAddresses = m_pMeta.m_AddressListEntries.size();

  const Tensor* input_input_t       = pOp.getInput(0);
  int32_t       input_input_ndim    = input_input_t->getNumOfDimensions();
  int32_t       input_input_dims[4] = {1, 1, 1, 1};
  for (int i = 0; i < input_input_ndim; ++i)
    input_input_dims[i] = input_input_t->dimension(i);
  int           input_mid = m_pMeta.getMemoryListEntryId(*input_input_t);
  NvDlaCubeInfo iinfo(*this, NVDLA_CUBE_FEATURE, input_input_dims[0], input_input_dims[1], input_input_dims[2],
                      input_input_dims[3]);

  const Tensor* output_output_t       = pOp.getOutput(0);
  int32_t       output_output_ndim    = output_output_t->getNumOfDimensions();
  int32_t       output_output_dims[4] = {1, 1, 1, 1};
  for (int i = 0; i < output_output_ndim; ++i)
    output_output_dims[i] = output_output_t->dimension(i);
  int           output_mid = m_pMeta.getMemoryListEntryId(*output_output_t);
  NvDlaCubeInfo oinfo(*this, NVDLA_CUBE_FEATURE, output_output_dims[0], output_output_dims[1], output_output_dims[2],
                      output_output_dims[3]);

  int32_t axis = pOp.getAxis().value();

  NvDlaEmuOperation*          softmax_op = new NvDlaEmuOperation();
  struct emu_softmax_op_desc* op_desc    = (struct emu_softmax_op_desc*)(&(softmax_op->op_desc));
  emu_common_op_desc          common;
  std::memset(&common, 0, sizeof(common));
  common.op_type             = NVDLA_EMU_OP_SOFTMAX;

  // The softmax runs on the CPU (the runtime's emulator), which reads its buffers by `format`:
  // FF16 = half floats (FP16 loadables); INT8/INT8_8 = int8 values converted with the two scale
  // factors (real score = int8 * input_scale_factor, int8 output = probability /
  // output_scale_factor). INT8 loadables used to say FF16 with zero scale factors, so the emulator
  // read the int8 scores two bytes at a time as half floats. As in nvdla_compiler, INT8_8 is used
  // when the memory atom is 8 bytes (nv_small), and the scales are the calibration scales of the
  // softmax's input and output tensors.
  NvU16 emuFormat = EMU_FORMAT_FF16;
  if (DLA_PRECISION == PRECISION_INT8) {
    emuFormat         = (FEATURE_ATOM_CUBE_SIZE == 8) ? EMU_FORMAT_INT8_8 : EMU_FORMAT_INT8;
    const float sfIn  = getActivationScale(input_input_t->getName());
    float       sfOut = getActivationScale(output_output_t->getName());
    if (sfIn <= 0.0f) {
      errs() << "[NvDla][int8] softmax " << output_output_t->getName() << " : no calibration scale for its input '"
             << input_input_t->getName() << "'; its output will be wrong\n";
    }
    if (sfOut <= 0.0f) {
      sfOut = 1.0f / 127.0f; // probabilities lie in [0, 1]
      outs() << "[NvDla][int8] softmax " << output_output_t->getName()
             << " : no calibration scale for its output, using 1/127\n";
    }
    common.input_scale_factor  = sfIn;
    common.output_scale_factor = sfOut;
    outs() << "[NvDla][int8] softmax " << output_output_t->getName() << " : SF_in=" << sfIn << " SF_out=" << sfOut
           << " emu format=" << emuFormat << "\n";
  }
  op_desc->axis              = axis;
  op_desc->common = common;

  struct emu_softmax_buffer_descs* op_buf = (struct emu_softmax_buffer_descs*)(&(softmax_op->op_buf));
  op_buf->src_data.addressIndex           = issueEmuAddr(input_mid);
  op_buf->src_data.size                   = m_pMeta.getMemoryListEntrySize(input_mid);
  op_buf->src_data.format                 = emuFormat;
  op_buf->src_data.width                  = input_input_dims[3];
  op_buf->src_data.height                 = input_input_dims[2];
  op_buf->src_data.channel                = input_input_dims[1];
  op_buf->src_data.line_stride            = iinfo.stride_line;
  op_buf->src_data.surf_stride            = iinfo.stride_surface;

  op_buf->dst_data.addressIndex = issueEmuAddr(output_mid);
  op_buf->dst_data.size         = m_pMeta.getMemoryListEntrySize(output_mid);
  op_buf->dst_data.format       = emuFormat;
  op_buf->dst_data.width        = output_output_dims[3];
  op_buf->dst_data.height       = output_output_dims[2];
  op_buf->dst_data.channel      = output_output_dims[1];
  op_buf->dst_data.line_stride  = oinfo.stride_line;
  op_buf->dst_data.surf_stride  = oinfo.stride_surface;

  issueEmuOp(softmax_op);
}
