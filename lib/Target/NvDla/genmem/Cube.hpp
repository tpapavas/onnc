#ifndef GENMEN_CUBE_H
#define GENMEN_CUBE_H

#include "Memory.hpp"
#include "header.hpp"

namespace genmem {

class Cube
{
public:
  enum class TYPE
  {
    CUBE,
    PLANAR
  };

  Cube(TYPE type, int w, int h, int c, int line, int surf, int planar, int batch_num = 1);
  Cube() {}

  string get_type() { return type == TYPE::CUBE ? "CUBE" : "PLANAR"; }
  void   allocate_memory(Memory& mem);
  void   allocate_memory_batched(Memory& mem);

  int used_entry_number() const;

  static int get_compact_line_stride(TYPE type, int W, int H, int C);
  static int get_compact_planar_stride(TYPE type, int W, int H, int C);
  static int get_compact_surface_stride(TYPE type, int W, int H, int C);

  static int get_compact_line_stride(TYPE type, const vector<int>& shape);
  static int get_compact_planar_stride(TYPE type, const vector<int>& shape);
  static int get_compact_surface_stride(TYPE type, const vector<int>& shape);

  TYPE type;
  int  W, H, C;
  int  line_stride, surf_stride, planar_stride;
  int  address_entry;
  int batch_num;
};
} // namespace genmem

#endif
