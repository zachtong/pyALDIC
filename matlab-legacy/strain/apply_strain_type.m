function [FStraintemp, FStrainWorld] = apply_strain_type(FSubpb2, DICpara)
%APPLY_STRAIN_TYPE  Convert infinitesimal strain to finite strain + world coords.
%
%   [FStraintemp, FStrainWorld] = apply_strain_type(FSubpb2, DICpara)
%
%   Inputs:
%     FSubpb2  - deformation gradient [4*nNodes x 1]: [F11,F21,F12,F22,...] per node
%     DICpara  - struct with .StrainType (0=infinitesimal, 1=Eulerian, 2=Green-Lagrangian)
%
%   Outputs:
%     FStraintemp  - strain after StrainType conversion [4*nNodes x 1]
%     FStrainWorld - world-coordinate strain (F21/F12 signs flipped)

FStrain = FSubpb2;
FStrainFinite = FStrain;

switch DICpara.StrainType
    case 0  % Infinitesimal — no-op
    case 1  % Eulerian-Almansi
        dudx = FStrain(1:4:end); dvdx = FStrain(2:4:end);
        dudy = FStrain(3:4:end); dvdy = FStrain(4:4:end);
        FStrainFinite(1:4:end) = 1./(1-dudx) - 1;
        FStrainFinite(4:4:end) = 1./(1-dvdy) - 1;
        FStrainFinite(3:4:end) = dudy./(1-dvdy);
        FStrainFinite(2:4:end) = dvdx./(1-dudx);
    case 2  % Green-Lagrangian
        dudx = FStrain(1:4:end); dvdx = FStrain(2:4:end);
        dudy = FStrain(3:4:end); dvdy = FStrain(4:4:end);
        FStrainFinite(1:4:end) = 0.5*(2*dudx - dudx.^2 - dvdx.^2);
        FStrainFinite(4:4:end) = 0.5*(2*dvdy - dudy.^2 - dvdy.^2);
        FStrainFinite(3:4:end) = 0.5*(dudy+dvdx - dudx.*dudy - dvdx.*dvdy);
        FStrainFinite(2:4:end) = 0.5*(dvdx+dudy - dudy.*dudx - dvdy.*dvdx);
    otherwise
        warning('apply_strain_type:unknown', 'Unknown StrainType %d, using infinitesimal.', DICpara.StrainType);
end

FStraintemp = FStrainFinite;
FStrainWorld = FStraintemp;
FStrainWorld(2:4:end) = -FStrainWorld(2:4:end);
FStrainWorld(3:4:end) = -FStrainWorld(3:4:end);

end
