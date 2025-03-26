// Repository: OpenHUTB/matlab
// File: toolbox/soc/fpga/target/+soc/+internal/genDesignTcl.m

function genDesignTcl(hbuild)
    switch hbuild.Vendor
    case 'Xilinx'
        soc.genXilinxDesignTcl(hbuild);
    case 'Intel'
        soc.genIntelDesignTcl(hbuild);
    end
end