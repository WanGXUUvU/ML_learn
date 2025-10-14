只有注册到 self._modules 里的子模块，PyTorch 才能自动管理它们的参数、递归调用 forward、设备迁移、保存/加载等。

net.state_dict() 保存了每个层的参数
