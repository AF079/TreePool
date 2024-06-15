
#include "header.h"

;


size_t get_num_class(void) {
  return 1;
}

size_t get_num_feature(void) {
  return 3;
}

const char* get_pred_transform(void) {
  return "identity";
}

float get_sigmoid_alpha(void) {
  return 1.0;
}

float get_ratio_c(void) {
  return 1.0;
}

float get_global_bias(void) {
  return 0.0;
}

const char* get_threshold_type(void) {
  return "float32";
}

const char* get_leaf_output_type(void) {
  return "float32";
}


static inline float pred_transform(float margin) {
  return margin;
}
float predict(union Entry* data, int pred_margin) {
  float sum = (float)0;
  unsigned int tmp;
  int nid, cond, fid;  /* used for folded subtrees */
  sum += predict_margin_unit0(data);

  sum = sum + (float)(0);
  if (!pred_margin) {
    return pred_transform(sum);
  } else {
    return sum;
  }
}
