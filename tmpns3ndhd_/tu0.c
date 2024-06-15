#include "header.h"
float predict_margin_unit0(union Entry* data) {
  float sum = (float)0;
  unsigned int tmp;
  int nid, cond, fid;  /* used for folded subtrees */
  if (!(data[0].missing != -1) || (data[0].fvalue < (float)5)) {
    sum += (float)0.60000002384;
  } else {
    sum += (float)-0.40000000596;
  }
  return sum;
}
