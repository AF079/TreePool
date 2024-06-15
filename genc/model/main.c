
#include "header.h"

;


size_t get_num_class(void) {
  return 3;
}

size_t get_num_feature(void) {
  return 4;
}

const char* get_pred_transform(void) {
  return "softmax";
}

float get_sigmoid_alpha(void) {
  return 1.0;
}

float get_ratio_c(void) {
  return 1.0;
}

float get_global_bias(void) {
  return 0.5;
}

const char* get_threshold_type(void) {
  return "float32";
}

const char* get_leaf_output_type(void) {
  return "float32";
}


static inline size_t pred_transform(float* pred) {
  const int num_class = 3;
  float max_margin = pred[0];
  double norm_const = 0.0;
  float t;
  for (int k = 1; k < num_class; ++k) {
    if (pred[k] > max_margin) {
      max_margin = pred[k];
    }
  }
  for (int k = 0; k < num_class; ++k) {
    t = expf(pred[k] - max_margin);
    norm_const += t;
    pred[k] = t;
  }
  for (int k = 0; k < num_class; ++k) {
    pred[k] /= (float)norm_const;
  }
  return (size_t)num_class;
}
size_t predict_multiclass(union Entry* data, int pred_margin, float* result) {
  float sum[3] = {0};
  unsigned int tmp;
  int nid, cond, fid;  /* used for folded subtrees */
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[0] += (float)0.14354066551;
  } else {
    sum[0] += (float)-0.073349639773;
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[1] += (float)-0.071770340204;
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
        sum[1] += (float)0.13880597055;
      } else {
        sum[1] += (float)-3.2511626724e-09;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.8500003815)) {
        sum[1] += (float)-2.5544848459e-09;
      } else {
        sum[1] += (float)-0.071270726621;
      }
    }
  }
  if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.6500000954)) {
    if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
      sum[2] += (float)-0.073299758136;
    } else {
      sum[2] += (float)0.072413794696;
    }
  } else {
    sum[2] += (float)0.13432835042;
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[0] += (float)0.12526625395;
  } else {
    sum[0] += (float)-0.070671133697;
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[1] += (float)-0.06909673661;
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
        sum[1] += (float)0.12130446732;
      } else {
        sum[1] += (float)0.00028283082065;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.8500003815)) {
        sum[1] += (float)0.0010258146795;
      } else {
        sum[1] += (float)-0.068723790348;
      }
    }
  }
  if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.6500000954)) {
    if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
      sum[2] += (float)-0.070614986122;
    } else {
      sum[2] += (float)0.067035496235;
    }
  } else {
    sum[2] += (float)0.1177168712;
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[0] += (float)0.11172717065;
  } else {
    sum[0] += (float)-0.068302184343;
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[1] += (float)-0.066703461111;
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
        sum[1] += (float)0.10821729153;
      } else {
        sum[1] += (float)0.00066645577317;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.8500003815)) {
        sum[1] += (float)0.0020745552611;
      } else {
        sum[1] += (float)-0.066424921155;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.75)) {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.5499999523)) {
      sum[2] += (float)-0.068131521344;
    } else {
      sum[2] += (float)0.0024170582183;
    }
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
        sum[2] += (float)-0.038548815995;
      } else {
        sum[2] += (float)0.043880090117;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.8500003815)) {
        sum[2] += (float)0.03168336302;
      } else {
        sum[2] += (float)0.11210051924;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[0] += (float)0.10134919733;
  } else {
    sum[0] += (float)-0.066190712154;
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[1] += (float)-0.064555004239;
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
        sum[1] += (float)0.098145976663;
      } else {
        sum[1] += (float)0.00055887090275;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.8500003815)) {
        sum[1] += (float)0.0018044585595;
      } else {
        sum[1] += (float)-0.064263910055;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.75)) {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.5499999523)) {
      sum[2] += (float)-0.066001549363;
    } else {
      sum[2] += (float)0.0035293155815;
    }
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
        sum[2] += (float)-0.037101786584;
      } else {
        sum[2] += (float)0.04062147066;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.8500003815)) {
        sum[2] += (float)0.029666190967;
      } else {
        sum[2] += (float)0.10161121935;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[0] += (float)0.093177855015;
  } else {
    sum[0] += (float)-0.064310312271;
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[1] += (float)-0.062621019781;
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
        sum[1] += (float)0.090139381588;
      } else {
        sum[1] += (float)0.00051908608293;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.8500003815)) {
        sum[1] += (float)0.0015939045697;
      } else {
        sum[1] += (float)-0.062313593924;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.75)) {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.5499999523)) {
      sum[2] += (float)-0.064097546041;
    } else {
      sum[2] += (float)0.0046413908713;
    }
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
        sum[2] += (float)-0.035708658397;
      } else {
        sum[2] += (float)0.03768613562;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.8500003815)) {
        sum[2] += (float)0.027812574059;
      } else {
        sum[2] += (float)0.093350499868;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[0] += (float)0.086605340242;
  } else {
    sum[0] += (float)-0.062630131841;
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[1] += (float)-0.060873564333;
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)5.0500001907)) {
        sum[1] += (float)0.082142442465;
      } else {
        sum[1] += (float)-0.011930166744;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.8500003815)) {
        sum[1] += (float)0.0014325578231;
      } else {
        sum[1] += (float)-0.060546755791;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.75)) {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.5499999523)) {
      sum[2] += (float)-0.062390804291;
    } else {
      sum[2] += (float)0.0057460065;
    }
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
        sum[2] += (float)-0.03436858952;
      } else {
        sum[2] += (float)0.035026583821;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.8500003815)) {
        sum[2] += (float)0.02610296011;
      } else {
        sum[2] += (float)0.086704097688;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[0] += (float)0.081224933267;
  } else {
    sum[0] += (float)-0.061123818159;
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[1] += (float)-0.059287782758;
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
        sum[1] += (float)0.078304901719;
      } else {
        sum[1] += (float)-0.00010426309746;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.8500003815)) {
        sum[1] += (float)0.0013117184862;
      } else {
        sum[1] += (float)-0.058939021081;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.75)) {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.5499999523)) {
      sum[2] += (float)-0.06086293608;
    } else {
      sum[2] += (float)0.0068053454161;
    }
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
        sum[2] += (float)-0.033098094165;
      } else {
        sum[2] += (float)0.033064834774;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.8500003815)) {
        sum[2] += (float)0.02452102676;
      } else {
        sum[2] += (float)0.081260919571;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[0] += (float)0.076754197478;
  } else {
    sum[0] += (float)-0.059767451137;
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[1] += (float)-0.05784162879;
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)5.0500001907)) {
        sum[1] += (float)0.072364084423;
      } else {
        sum[1] += (float)-0.01144558005;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.8500003815)) {
        sum[1] += (float)0.0012240804499;
      } else {
        sum[1] += (float)-0.057468678802;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.75)) {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.5499999523)) {
      sum[2] += (float)-0.059476818889;
    } else {
      sum[2] += (float)0.0078795216978;
    }
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
        sum[2] += (float)-0.031858492643;
      } else {
        sum[2] += (float)0.030816301703;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.8500003815)) {
        sum[2] += (float)0.023053031415;
      } else {
        sum[2] += (float)0.076735623181;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[0] += (float)0.072991169989;
  } else {
    sum[0] += (float)-0.058542318642;
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[1] += (float)-0.05651557073;
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
        sum[1] += (float)0.069942377508;
      } else {
        sum[1] += (float)-0.000462966942;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.8500003815)) {
        sum[1] += (float)0.001163502573;
      } else {
        sum[1] += (float)-0.056116428226;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.75)) {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.5499999523)) {
      sum[2] += (float)-0.058226455003;
    } else {
      sum[2] += (float)0.0088995676488;
    }
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)5.0500001907)) {
        sum[2] += (float)-0.017864815891;
      } else {
        sum[2] += (float)0.036388445646;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.8500003815)) {
        sum[2] += (float)0.021687334403;
      } else {
        sum[2] += (float)0.072923965752;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[0] += (float)0.069787405431;
  } else {
    sum[0] += (float)-0.057432204485;
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[1] += (float)-0.055292338133;
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
        sum[1] += (float)0.066606275737;
      } else {
        sum[1] += (float)-0.00058142642956;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.8500003815)) {
        sum[1] += (float)0.001124826842;
      } else {
        sum[1] += (float)-0.054865192622;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.75)) {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.5499999523)) {
      sum[2] += (float)-0.057081993669;
    } else {
      sum[2] += (float)0.0099280122668;
    }
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)5.0500001907)) {
        sum[2] += (float)-0.01710879989;
      } else {
        sum[2] += (float)0.034402985126;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.8500003815)) {
        sum[2] += (float)0.020414028317;
      } else {
        sum[2] += (float)0.069675944746;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[0] += (float)0.067031532526;
  } else {
    sum[0] += (float)-0.056419860572;
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[1] += (float)-0.054156657308;
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
        sum[1] += (float)0.063686363399;
      } else {
        sum[1] += (float)-0.00067048531491;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.8500003815)) {
        sum[1] += (float)0.0011037138756;
      } else {
        sum[1] += (float)-0.05369983241;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.75)) {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.5499999523)) {
      sum[2] += (float)-0.05603460595;
    } else {
      sum[2] += (float)0.01092935726;
    }
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)5.0500001907)) {
        sum[2] += (float)-0.016376242042;
      } else {
        sum[2] += (float)0.032558836043;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.8500003815)) {
        sum[2] += (float)0.019224589691;
      } else {
        sum[2] += (float)0.066878937185;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[0] += (float)0.064638279378;
  } else {
    sum[0] += (float)-0.055491894484;
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[1] += (float)-0.053094994277;
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.6500000954)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
        sum[1] += (float)0.065591804683;
      } else {
        sum[1] += (float)-0.020583773032;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)5.0500001907)) {
        sum[1] += (float)-0.013719475828;
      } else {
        sum[1] += (float)-0.051988966763;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.75)) {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.5499999523)) {
      sum[2] += (float)-0.055070769042;
    } else {
      sum[2] += (float)0.011901489459;
    }
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.5499999523)) {
        sum[2] += (float)0.029638716951;
      } else {
        sum[2] += (float)-0.020656151697;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.8500003815)) {
        sum[2] += (float)0.01811167784;
      } else {
        sum[2] += (float)0.064446754754;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[0] += (float)0.062541179359;
  } else {
    sum[0] += (float)-0.054634392262;
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[1] += (float)-0.052095402032;
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.6500000954)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
        sum[1] += (float)0.063474521041;
      } else {
        sum[1] += (float)-0.020141631365;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)5.0500001907)) {
        sum[1] += (float)-0.012912412174;
      } else {
        sum[1] += (float)-0.050933934748;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.75)) {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.5499999523)) {
      sum[2] += (float)-0.05416277051;
    } else {
      sum[2] += (float)0.012211638503;
    }
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.5499999523)) {
        sum[2] += (float)0.028591850773;
      } else {
        sum[2] += (float)-0.020943818614;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.8500003815)) {
        sum[2] += (float)0.016771527007;
      } else {
        sum[2] += (float)0.062357317656;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[0] += (float)0.060687869787;
  } else {
    sum[0] += (float)-0.053840983659;
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[1] += (float)-0.051147233695;
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)5.0500001907)) {
        sum[1] += (float)0.055305369198;
      } else {
        sum[1] += (float)-0.009413190186;
      }
    } else {
      if (!(data[0].missing != -1) || (data[0].fvalue < (float)5.9499998093)) {
        sum[1] += (float)-0.00068187306169;
      } else {
        sum[1] += (float)-0.052016824484;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.75)) {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.5499999523)) {
      sum[2] += (float)-0.053316473961;
    } else {
      sum[2] += (float)0.012512509711;
    }
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.5499999523)) {
        sum[2] += (float)0.027619540691;
      } else {
        sum[2] += (float)-0.021194554865;
      }
    } else {
      if (!(data[0].missing != -1) || (data[0].fvalue < (float)5.9499998093)) {
        sum[2] += (float)0.022845244035;
      } else {
        sum[2] += (float)0.062085092068;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[0] += (float)0.059036221355;
  } else {
    sum[0] += (float)-0.05311184749;
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[1] += (float)-0.050241108984;
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.6500000954)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
        sum[1] += (float)0.060048319399;
      } else {
        sum[1] += (float)-0.020008157939;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)5.0500001907)) {
        sum[1] += (float)-0.011407775804;
      } else {
        sum[1] += (float)-0.049017868936;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.75)) {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.5499999523)) {
      sum[2] += (float)-0.052542518824;
    } else {
      sum[2] += (float)0.013362389989;
    }
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.5499999523)) {
        sum[2] += (float)0.027136636898;
      } else {
        sum[2] += (float)-0.020782565698;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.8500003815)) {
        sum[2] += (float)0.012913499959;
      } else {
        sum[2] += (float)0.058919388801;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[0] += (float)0.057551927865;
  } else {
    sum[0] += (float)-0.052416980267;
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[1] += (float)-0.049368705601;
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.6500000954)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
        sum[1] += (float)0.058559726924;
      } else {
        sum[1] += (float)-0.019549364224;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)5.0500001907)) {
        sum[1] += (float)-0.010708465241;
      } else {
        sum[1] += (float)-0.048077087849;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.75)) {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.5499999523)) {
      sum[2] += (float)-0.05179027468;
    } else {
      sum[2] += (float)0.01364074368;
    }
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.5499999523)) {
        sum[2] += (float)0.026267454028;
      } else {
        sum[2] += (float)-0.020988104865;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.8500003815)) {
        sum[2] += (float)0.011848266236;
      } else {
        sum[2] += (float)0.057422716171;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[0] += (float)0.056207116693;
  } else {
    sum[0] += (float)-0.051760978997;
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[1] += (float)-0.04852264002;
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.6500000954)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
        sum[1] += (float)0.057218458503;
      } else {
        sum[1] += (float)-0.019101379439;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)5.0500001907)) {
        sum[1] += (float)-0.010041375645;
      } else {
        sum[1] += (float)-0.047160431743;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.75)) {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.5499999523)) {
      sum[2] += (float)-0.051073618233;
    } else {
      sum[2] += (float)0.01390811801;
    }
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.5499999523)) {
        sum[2] += (float)0.025456273928;
      } else {
        sum[2] += (float)-0.021162224934;
      }
    } else {
      if (!(data[0].missing != -1) || (data[0].fvalue < (float)5.9499998093)) {
        sum[2] += (float)0.016588483006;
      } else {
        sum[2] += (float)0.057583451271;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[0] += (float)0.054978560656;
  } else {
    sum[0] += (float)-0.051138933748;
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[1] += (float)-0.047696430236;
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.6500000954)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
        sum[1] += (float)0.056000646204;
      } else {
        sum[1] += (float)-0.018664980307;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)5.0500001907)) {
        sum[1] += (float)-0.0091555807739;
      } else {
        sum[1] += (float)-0.046291701496;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.75)) {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.5499999523)) {
      sum[2] += (float)-0.050385512412;
    } else {
      sum[2] += (float)0.014164266177;
    }
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.75)) {
      if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.5499999523)) {
        sum[2] += (float)0.024698046967;
      } else {
        sum[2] += (float)-0.021306723356;
      }
    } else {
      if (!(data[0].missing != -1) || (data[0].fvalue < (float)5.9499998093)) {
        sum[2] += (float)0.015042501502;
      } else {
        sum[2] += (float)0.056285142899;
      }
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[0] += (float)0.053846854717;
  } else {
    sum[0] += (float)-0.05054352805;
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[1] += (float)-0.046884387732;
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.6500000954)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
        sum[1] += (float)0.054886430502;
      } else {
        sum[1] += (float)-0.018240755424;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)5.0500001907)) {
        sum[1] += (float)-0.0083051444963;
      } else {
        sum[1] += (float)-0.045438464731;
      }
    }
  }
  if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.6500000954)) {
    if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
      sum[2] += (float)-0.050712417811;
    } else {
      sum[2] += (float)0.0340141505;
    }
  } else {
    if (!(data[2].missing != -1) || (data[2].fvalue < (float)5.0500001907)) {
      if (!(data[1].missing != -1) || (data[1].fvalue < (float)2.9000000954)) {
        sum[2] += (float)0.056709509343;
      } else {
        sum[2] += (float)-0.01885353215;
      }
    } else {
      sum[2] += (float)0.052951302379;
    }
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[0] += (float)0.052790757269;
  } else {
    sum[0] += (float)-0.049956053495;
  }
  if (!(data[2].missing != -1) || (data[2].fvalue < (float)2.4500000477)) {
    sum[1] += (float)-0.046082153916;
  } else {
    if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.6500000954)) {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
        sum[1] += (float)0.053716149181;
      } else {
        sum[1] += (float)-0.017123175785;
      }
    } else {
      if (!(data[2].missing != -1) || (data[2].fvalue < (float)5.0500001907)) {
        sum[1] += (float)-0.0076414393261;
      } else {
        sum[1] += (float)-0.044567551464;
      }
    }
  }
  if (!(data[3].missing != -1) || (data[3].fvalue < (float)1.6500000954)) {
    if (!(data[2].missing != -1) || (data[2].fvalue < (float)4.9499998093)) {
      sum[2] += (float)-0.050039298832;
    } else {
      sum[2] += (float)0.032058633864;
    }
  } else {
    if (!(data[2].missing != -1) || (data[2].fvalue < (float)5.0500001907)) {
      if (!(data[1].missing != -1) || (data[1].fvalue < (float)2.9000000954)) {
        sum[2] += (float)0.055208563805;
      } else {
        sum[2] += (float)-0.018903901801;
      }
    } else {
      sum[2] += (float)0.051836509258;
    }
  }

  for (int i = 0; i < 3; ++i) {
    result[i] = sum[i] + (float)(0.5);
  }
  if (!pred_margin) {
    return pred_transform(result);
  } else {
    return 3;
  }
}
