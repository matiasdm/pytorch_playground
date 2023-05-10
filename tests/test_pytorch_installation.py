def test_pythorch_installation():
    import torch 
    x = torch.ones(1, 1)
    assert x[0] == 1  # check torch is available and can create a trivial tensor
    assert torch.cuda.is_available()  # check we have access to cuda 
