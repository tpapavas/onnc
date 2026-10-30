PP_DEFINE_VISIT(Add, pAdd)
{
  NvDlaBackendMeta::OpEngineMapper::Category engine_category =
      m_pMeta.m_OpEngineMapper.getAndSetOpEngine(m_pMeta.m_OpEngineMapper,
                                                 NvDlaBackendMeta::OpEngineMapper::NvDlaOpType::sdp);
  outs() << "[MAPPING] op engine: " << static_cast<std::underlying_type<NvDlaBackendMeta::OpEngineMapper::Category>::type>(engine_category) << "\n";
  if (engine_category == NvDlaBackendMeta::OpEngineMapper::Category::emu) {
    PP_CALL_VISIT_CPU(pAdd);

    return;
  }

  emitSdp(pAdd, *pAdd.getA(), *pAdd.getB(), *pAdd.getOutput(0));
}
