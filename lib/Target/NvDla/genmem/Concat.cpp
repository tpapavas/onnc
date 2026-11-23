#include "Operators.hpp"

#include <iostream>
#include <vector>
#include <onnc/Support/IOStream.h>

using namespace genmem;

void Concat::concat(vector<Cube>& in_cube, Cube& out_cube, Memory& mem, vector<Register>& regs, int axis,
                    int input_tensor_num)
{
  switch (axis) {
  case 0:
    concat_axis_0(in_cube, out_cube, mem, regs, input_tensor_num);
    break;
  }
}

void Concat::concat_axis_0(vector<Cube>& input_cube, Cube& output_cube, Memory& mem, vector<Register>& regs,
                           int input_tensor_num)
{
  vector<vector<Register>> input_cubes;
  vector<Register> output_cubes;
  Cube before_merge_cube(
    Cube::TYPE::PLANAR, output_cube.W, output_cube.H, output_cube.C,
    Cube::get_compact_line_stride(Cube::TYPE::PLANAR, output_cube.W, output_cube.H, output_cube.C), -1,
    Cube::get_compact_planar_stride(Cube::TYPE::PLANAR, output_cube.W, output_cube.H, output_cube.C), output_cube.batch_num);
  before_merge_cube.allocate_memory_batched(mem);

//   onnc::outs() << "[RUBIK] batches: " << output_cube.batch_num << "\n";
  for (int nn = 0; nn < output_cube.batch_num; nn++) {
  Cube bm_cube = before_merge_cube;
  bm_cube.address_entry += nn * before_merge_cube.used_entry_number();
  // split
  for (int i = 0; i < input_tensor_num; i++) {
    Cube in_cube = input_cube[i];
    in_cube.address_entry += nn * input_cube[i].used_entry_number();
    // GEN_assert(input_cube[i].address_entry <= input_cube[i].used_entry_number()*input_cube[i].batch_num);

    // output planar
    Cube out_cube_split(
      Cube::TYPE::PLANAR, input_cube[i].W, input_cube[i].H, input_cube[i].C,
      Cube::get_compact_line_stride(Cube::TYPE::PLANAR, input_cube[i].W, input_cube[i].H, input_cube[i].C), -1,
      Cube::get_compact_planar_stride(Cube::TYPE::PLANAR, input_cube[i].W, input_cube[i].H, input_cube[i].C));
    GEN_assert(before_merge_cube.planar_stride == out_cube_split.planar_stride);
    GEN_assert(before_merge_cube.line_stride == out_cube_split.line_stride);

    int total_planar_used = 0;
    for (int j = 0; j < i; j++) {
      total_planar_used += input_cube[j].C;
    }
    out_cube_split.address_entry = bm_cube.address_entry + total_planar_used * out_cube_split.planar_stride;

    // split
    Register reg;
    // Split::split(input_cube[i], out_cube_split, mem, reg);
    Split::split(in_cube, out_cube_split, mem, reg);
    // regs.push_back(reg);
    if (nn == 0) {
        vector<Register> reg_v;
        input_cubes.push_back(reg_v);
    }
    input_cubes[i].push_back(reg);

  }
  Cube out_cube = output_cube;
  out_cube.address_entry += nn * output_cube.used_entry_number();

  // merge
  Register reg;
//   Merge::merge(before_merge_cube, output_cube, mem, reg);
  Merge::merge(bm_cube, out_cube, mem, reg);
//   regs.push_back(reg);
  output_cubes.push_back(reg);

  } // BATCHES

  /**
   * push input regs by input id
   * E.g. for 2 inputs and 2 batches:
   * i0b0 -> i0b1 -> i1b0 -> i1b1
   * (we want a graph op -e.g. split here- to be registered for all batches
   *  -that is, 1st split for batch 0, then 1st split for batch 1 etc-
   *  before moving to the next graph op -that is, 2nd split for batch 0,
   *  2nd split for batch 1 etc)
   */
  for (auto& row : input_cubes) {
    for (Register& reg : row) {
        regs.push_back(reg);
    }
  }

  for (Register& reg : output_cubes) {
    regs.push_back(reg);
  }
}
