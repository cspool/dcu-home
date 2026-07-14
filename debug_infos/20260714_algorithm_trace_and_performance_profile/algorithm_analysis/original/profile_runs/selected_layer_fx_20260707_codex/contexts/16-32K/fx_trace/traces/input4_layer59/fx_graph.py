


def forward(self, arg0_1, arg1_1, arg2_1):
    _param_constant0 = self._param_constant0
    detach = torch.ops.aten.detach.default(_param_constant0);  _param_constant0 = None
    add = torch.ops.aten.add.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
    _to_copy = torch.ops.aten._to_copy.default(add, dtype = torch.float32)
    pow_1 = torch.ops.aten.pow.Tensor_Scalar(_to_copy, 2)
    mean = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
    add_1 = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
    rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    mul = torch.ops.aten.mul.Tensor(_to_copy, rsqrt);  _to_copy = rsqrt = None
    _to_copy_1 = torch.ops.aten._to_copy.default(detach, dtype = torch.float32);  detach = None
    add_2 = torch.ops.aten.add.Tensor(_to_copy_1, 1.0);  _to_copy_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(mul, add_2);  mul = add_2 = None
    _to_copy_2 = torch.ops.aten._to_copy.default(mul_1, dtype = torch.bfloat16);  mul_1 = None
    empty_like = torch.ops.aten.empty_like.default(_to_copy_2, pin_memory = False)
    _param_constant1 = self._param_constant1
    t = torch.ops.aten.t.default(_param_constant1);  _param_constant1 = None
    mm = torch.ops.aten.mm.default(_to_copy_2, t);  _to_copy_2 = t = None
    split_with_sizes = torch.ops.aten.split_with_sizes.default(mm, [12288, 1024, 1024], -1);  mm = None
    getitem = split_with_sizes[0]
    getitem_1 = split_with_sizes[1]
    getitem_2 = split_with_sizes[2];  split_with_sizes = None
    view = torch.ops.aten.view.default(getitem, [4096, 24, -1]);  getitem = None
    split = torch.ops.aten.split.Tensor(view, 256, -1);  view = None
    getitem_3 = split[0]
    getitem_4 = split[1];  split = None
    clone = torch.ops.aten.clone.default(getitem_3, memory_format = torch.contiguous_format);  getitem_3 = None
    _unsafe_view = torch.ops.aten._unsafe_view.default(clone, [4096, 6144]);  clone = None
    clone_1 = torch.ops.aten.clone.default(getitem_4, memory_format = torch.contiguous_format);  getitem_4 = None
    _unsafe_view_1 = torch.ops.aten._unsafe_view.default(clone_1, [4096, 6144]);  clone_1 = None
    view_1 = torch.ops.aten.view.default(_unsafe_view, [-1, 24, 256]);  _unsafe_view = None
    _param_constant2 = self._param_constant2
    detach_1 = torch.ops.aten.detach.default(_param_constant2);  _param_constant2 = None
    _to_copy_3 = torch.ops.aten._to_copy.default(view_1, dtype = torch.float32);  view_1 = None
    pow_2 = torch.ops.aten.pow.Tensor_Scalar(_to_copy_3, 2)
    mean_1 = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
    add_3 = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
    rsqrt_1 = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    mul_2 = torch.ops.aten.mul.Tensor(_to_copy_3, rsqrt_1);  _to_copy_3 = rsqrt_1 = None
    _to_copy_4 = torch.ops.aten._to_copy.default(detach_1, dtype = torch.float32);  detach_1 = None
    add_4 = torch.ops.aten.add.Tensor(_to_copy_4, 1.0);  _to_copy_4 = None
    mul_3 = torch.ops.aten.mul.Tensor(mul_2, add_4);  mul_2 = add_4 = None
    _to_copy_5 = torch.ops.aten._to_copy.default(mul_3, dtype = torch.bfloat16);  mul_3 = None
    view_2 = torch.ops.aten.view.default(_to_copy_5, [-1, 6144]);  _to_copy_5 = None
    view_3 = torch.ops.aten.view.default(getitem_1, [-1, 4, 256]);  getitem_1 = None
    _param_constant3 = self._param_constant3
    detach_2 = torch.ops.aten.detach.default(_param_constant3);  _param_constant3 = None
    _to_copy_6 = torch.ops.aten._to_copy.default(view_3, dtype = torch.float32);  view_3 = None
    pow_3 = torch.ops.aten.pow.Tensor_Scalar(_to_copy_6, 2)
    mean_2 = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
    add_5 = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
    rsqrt_2 = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
    mul_4 = torch.ops.aten.mul.Tensor(_to_copy_6, rsqrt_2);  _to_copy_6 = rsqrt_2 = None
    _to_copy_7 = torch.ops.aten._to_copy.default(detach_2, dtype = torch.float32);  detach_2 = None
    add_6 = torch.ops.aten.add.Tensor(_to_copy_7, 1.0);  _to_copy_7 = None
    mul_5 = torch.ops.aten.mul.Tensor(mul_4, add_6);  mul_4 = add_6 = None
    _to_copy_8 = torch.ops.aten._to_copy.default(mul_5, dtype = torch.bfloat16);  mul_5 = None
    view_4 = torch.ops.aten.view.default(_to_copy_8, [-1, 1024]);  _to_copy_8 = None
    _tensor_constant0 = self._tensor_constant0
    index = torch.ops.aten.index.Tensor(_tensor_constant0, [arg0_1]);  _tensor_constant0 = arg0_1 = None
    split_1 = torch.ops.aten.split.Tensor(index, 32, -1);  index = None
    getitem_5 = split_1[0]
    getitem_6 = split_1[1];  split_1 = None
    select = torch.ops.aten.select.int(getitem_5, 0, 0)
    clone_2 = torch.ops.aten.clone.default(select);  select = None
    select_1 = torch.ops.aten.select.int(getitem_5, 0, 1)
    slice_1 = torch.ops.aten.slice.Tensor(select_1, 1, 1, 33, 3);  select_1 = None
    slice_2 = torch.ops.aten.slice.Tensor(clone_2, 1, 1, 33, 3)
    copy_ = torch.ops.aten.copy_.default(slice_2, slice_1);  slice_2 = slice_1 = copy_ = None
    select_2 = torch.ops.aten.select.int(getitem_5, 0, 2);  getitem_5 = None
    slice_3 = torch.ops.aten.slice.Tensor(select_2, 1, 2, 30, 3);  select_2 = None
    slice_4 = torch.ops.aten.slice.Tensor(clone_2, 1, 2, 30, 3)
    copy__1 = torch.ops.aten.copy_.default(slice_4, slice_3);  slice_4 = slice_3 = copy__1 = None
    select_3 = torch.ops.aten.select.int(getitem_6, 0, 0)
    clone_3 = torch.ops.aten.clone.default(select_3);  select_3 = None
    select_4 = torch.ops.aten.select.int(getitem_6, 0, 1)
    slice_5 = torch.ops.aten.slice.Tensor(select_4, 1, 1, 33, 3);  select_4 = None
    slice_6 = torch.ops.aten.slice.Tensor(clone_3, 1, 1, 33, 3)
    copy__2 = torch.ops.aten.copy_.default(slice_6, slice_5);  slice_6 = slice_5 = copy__2 = None
    select_5 = torch.ops.aten.select.int(getitem_6, 0, 2);  getitem_6 = None
    slice_7 = torch.ops.aten.slice.Tensor(select_5, 1, 2, 30, 3);  select_5 = None
    slice_8 = torch.ops.aten.slice.Tensor(clone_3, 1, 2, 30, 3)
    copy__3 = torch.ops.aten.copy_.default(slice_8, slice_7);  slice_8 = slice_7 = copy__3 = None
    view_5 = torch.ops.aten.view.default(view_2, [4096, -1, 256]);  view_2 = None
    slice_9 = torch.ops.aten.slice.Tensor(view_5, 2, 0, 64)
    slice_10 = torch.ops.aten.slice.Tensor(view_5, 2, 64, 9223372036854775807);  view_5 = None
    unsqueeze = torch.ops.aten.unsqueeze.default(clone_2, -2)
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(clone_3, -2)
    split_2 = torch.ops.aten.split.Tensor(slice_9, 32, -1);  slice_9 = None
    getitem_7 = split_2[0]
    getitem_8 = split_2[1];  split_2 = None
    mul_6 = torch.ops.aten.mul.Tensor(getitem_7, unsqueeze)
    mul_7 = torch.ops.aten.mul.Tensor(getitem_8, unsqueeze_1)
    sub = torch.ops.aten.sub.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
    mul_8 = torch.ops.aten.mul.Tensor(getitem_8, unsqueeze);  getitem_8 = unsqueeze = None
    mul_9 = torch.ops.aten.mul.Tensor(getitem_7, unsqueeze_1);  getitem_7 = unsqueeze_1 = None
    add_7 = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
    cat = torch.ops.aten.cat.default([sub, add_7], -1);  sub = add_7 = None
    cat_1 = torch.ops.aten.cat.default([cat, slice_10], -1);  cat = slice_10 = None
    view_6 = torch.ops.aten.view.default(cat_1, [4096, 6144]);  cat_1 = None
    view_7 = torch.ops.aten.view.default(view_4, [4096, -1, 256]);  view_4 = None
    slice_11 = torch.ops.aten.slice.Tensor(view_7, 2, 0, 64)
    slice_12 = torch.ops.aten.slice.Tensor(view_7, 2, 64, 9223372036854775807);  view_7 = None
    unsqueeze_2 = torch.ops.aten.unsqueeze.default(clone_2, -2);  clone_2 = None
    unsqueeze_3 = torch.ops.aten.unsqueeze.default(clone_3, -2);  clone_3 = None
    split_3 = torch.ops.aten.split.Tensor(slice_11, 32, -1);  slice_11 = None
    getitem_9 = split_3[0]
    getitem_10 = split_3[1];  split_3 = None
    mul_10 = torch.ops.aten.mul.Tensor(getitem_9, unsqueeze_2)
    mul_11 = torch.ops.aten.mul.Tensor(getitem_10, unsqueeze_3)
    sub_1 = torch.ops.aten.sub.Tensor(mul_10, mul_11);  mul_10 = mul_11 = None
    mul_12 = torch.ops.aten.mul.Tensor(getitem_10, unsqueeze_2);  getitem_10 = unsqueeze_2 = None
    mul_13 = torch.ops.aten.mul.Tensor(getitem_9, unsqueeze_3);  getitem_9 = unsqueeze_3 = None
    add_8 = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
    cat_2 = torch.ops.aten.cat.default([sub_1, add_8], -1);  sub_1 = add_8 = None
    cat_3 = torch.ops.aten.cat.default([cat_2, slice_12], -1);  cat_2 = slice_12 = None
    view_8 = torch.ops.aten.view.default(cat_3, [4096, 1024]);  cat_3 = None
    empty = torch.ops.aten.empty.memory_format([4096, 6144], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
    view_9 = torch.ops.aten.view.default(view_6, [-1, 24, 256]);  view_6 = None
    view_10 = torch.ops.aten.view.default(empty, [-1, 24, 256]);  empty = None
    view_11 = torch.ops.aten.view.default(view_8, [-1, 4, 256]);  view_8 = None
    view_12 = torch.ops.aten.view.default(getitem_2, [-1, 4, 256]);  getitem_2 = None
    unified_kv_cache_update = torch.ops.vllm.unified_kv_cache_update.default(view_11, view_12, 'language_model.model.layers.59.self_attn.attn')
    unified_attention_with_output = torch.ops.vllm.unified_attention_with_output.default(view_9, view_11, view_12, view_10, 'language_model.model.layers.59.self_attn.attn', None, None, unified_kv_cache_update);  view_9 = view_11 = view_12 = unified_kv_cache_update = unified_attention_with_output = None
    view_13 = torch.ops.aten.view.default(view_10, [-1, 6144]);  view_10 = None
    sigmoid = torch.ops.aten.sigmoid.default(_unsafe_view_1);  _unsafe_view_1 = None
    mul_14 = torch.ops.aten.mul.Tensor(view_13, sigmoid);  view_13 = sigmoid = None
    _param_constant4 = self._param_constant4
    t_1 = torch.ops.aten.t.default(_param_constant4);  _param_constant4 = None
    mm_1 = torch.ops.aten.mm.default(mul_14, t_1);  mul_14 = t_1 = None
    copy__4 = torch.ops.aten.copy_.default(empty_like, mm_1);  empty_like = mm_1 = None
    _param_constant5 = self._param_constant5
    detach_3 = torch.ops.aten.detach.default(_param_constant5);  _param_constant5 = None
    add_9 = torch.ops.aten.add.Tensor(copy__4, add);  copy__4 = add = None
    _to_copy_9 = torch.ops.aten._to_copy.default(add_9, dtype = torch.float32)
    pow_4 = torch.ops.aten.pow.Tensor_Scalar(_to_copy_9, 2)
    mean_3 = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
    add_10 = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
    rsqrt_3 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    mul_15 = torch.ops.aten.mul.Tensor(_to_copy_9, rsqrt_3);  _to_copy_9 = rsqrt_3 = None
    _to_copy_10 = torch.ops.aten._to_copy.default(detach_3, dtype = torch.float32);  detach_3 = None
    add_11 = torch.ops.aten.add.Tensor(_to_copy_10, 1.0);  _to_copy_10 = None
    mul_16 = torch.ops.aten.mul.Tensor(mul_15, add_11);  mul_15 = add_11 = None
    _to_copy_11 = torch.ops.aten._to_copy.default(mul_16, dtype = torch.bfloat16);  mul_16 = None
    _param_constant6 = self._param_constant6
    t_2 = torch.ops.aten.t.default(_param_constant6);  _param_constant6 = None
    mm_2 = torch.ops.aten.mm.default(_to_copy_11, t_2);  _to_copy_11 = t_2 = None
    empty_1 = torch.ops.aten.empty.memory_format([4096, 17408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
    silu_and_mul = torch.ops._C.silu_and_mul.default(empty_1, mm_2);  mm_2 = silu_and_mul = None
    _param_constant7 = self._param_constant7
    t_3 = torch.ops.aten.t.default(_param_constant7);  _param_constant7 = None
    mm_3 = torch.ops.aten.mm.default(empty_1, t_3);  empty_1 = t_3 = None
    return (mm_3, add_9)
    
