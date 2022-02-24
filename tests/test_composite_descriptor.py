from optunaz.descriptors import CompositeDescriptor, ECFP, PhyschemDescriptors


def test_composite():
    d1 = ECFP.new(nBits=1024)
    d2 = PhyschemDescriptors.new()
    d_comp = CompositeDescriptor.new(descriptors=[d1, d2])

    smi = "CCC"

    fp1 = d1.calculate_from_smi(smi)
    fp2 = d2.calculate_from_smi(smi)
    fp_comp = d_comp.calculate_from_smi(smi)

    assert len(fp1) == 1024
    assert len(fp2) == 208
    assert len(fp_comp) == 1232
