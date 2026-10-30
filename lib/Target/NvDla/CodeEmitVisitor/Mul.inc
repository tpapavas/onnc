PP_DEFINE_VISIT(Mul, pMul)
{
  NvDlaBackendMeta::OpEngineMapper::Category engine_category =
      m_pMeta.m_OpEngineMapper.getAndSetOpEngine(m_pMeta.m_OpEngineMapper,
                                                 NvDlaBackendMeta::OpEngineMapper::NvDlaOpType::sdp);
  outs() << "[MAPPING] op engine: " << static_cast<std::underlying_type<NvDlaBackendMeta::OpEngineMapper::Category>::type>(engine_category) << "\n";
  if (engine_category == NvDlaBackendMeta::OpEngineMapper::Category::emu) {
    PP_CALL_VISIT_CPU(pMul);

    return;
  }

  emitSdp(pMul, *pMul.getA(), *pMul.getB(), *pMul.getOutput(0));
}
