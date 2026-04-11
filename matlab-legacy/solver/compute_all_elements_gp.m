function [N_all, DN_all, Jdet_all] = compute_all_elements_gp( ...
        ksi, eta, ptx, pty, delta, nEle)
% Compute shape functions, DN matrix, and Jacobian determinant for all
% elements at one Gauss point (ksi, eta).
%
% Supports Q4 elements with optional hanging-node midpoints (8-node).
%
% Inputs:
%   ksi, eta   - scalar Gauss point in reference coords
%   ptx, pty   - nEle x 8, element node coordinates
%   delta      - nEle x 4, midpoint flags for nodes 5-8
%   nEle       - number of elements
%
% Outputs:
%   N_all    - 2 x 16 x nEle, shape function matrix
%   DN_all   - 4 x 16 x nEle, shape function gradient matrix (physical)
%   Jdet_all - nEle x 1, Jacobian determinants

d5 = delta(:,1); d6 = delta(:,2); d7 = delta(:,3); d8 = delta(:,4);

% --- Shape functions (nEle x 1 each) ---
N5 = d5 .* (0.5*(1+ksi)*(1-abs(eta)));
N6 = d6 .* (0.5*(1+eta)*(1-abs(ksi)));
N7 = d7 .* (0.5*(1-ksi)*(1-abs(eta)));
N8 = d8 .* (0.5*(1-eta)*(1-abs(ksi)));
N1 = (1-ksi)*(1-eta)*0.25 - 0.5*(N7+N8);
N2 = (1+ksi)*(1-eta)*0.25 - 0.5*(N8+N5);
N3 = (1+ksi)*(1+eta)*0.25 - 0.5*(N5+N6);
N4 = (1-ksi)*(1+eta)*0.25 - 0.5*(N6+N7);

% --- Shape function derivatives w.r.t. ksi (nEle x 1 each) ---
seta = sign(-eta); sksi = sign(-ksi);
dN5k = d5 .* (0.5*(1-abs(eta)));
dN6k = d6 .* (0.5*(1+eta)*sksi);
dN7k = d7 .* (-0.5*(1-abs(eta)));
dN8k = d8 .* (0.5*(1-eta)*sksi);
dN1k = -0.25*(1-eta) - 0.5*(dN7k + dN8k);
dN2k =  0.25*(1-eta) - 0.5*(dN8k + dN5k);
dN3k =  0.25*(1+eta) - 0.5*(dN5k + dN6k);
dN4k = -0.25*(1+eta) - 0.5*(dN6k + dN7k);

% --- Shape function derivatives w.r.t. eta (nEle x 1 each) ---
dN5e = d5 .* (0.5*(1+ksi)*seta);
dN6e = d6 .* (0.5*(1-abs(ksi)));
dN7e = d7 .* (0.5*(1-ksi)*seta);
dN8e = d8 .* (-0.5*(1-abs(ksi)));
dN1e = -0.25*(1-ksi) - 0.5*(dN7e + dN8e);
dN2e = -0.25*(1+ksi) - 0.5*(dN8e + dN5e);
dN3e =  0.25*(1+ksi) - 0.5*(dN5e + dN6e);
dN4e =  0.25*(1-ksi) - 0.5*(dN6e + dN7e);

% Stack dN/dksi and dN/deta: nEle x 8
dNdk = [dN1k, dN2k, dN3k, dN4k, dN5k, dN6k, dN7k, dN8k];
dNde = [dN1e, dN2e, dN3e, dN4e, dN5e, dN6e, dN7e, dN8e];

% --- Jacobian: J = [J11 J12; J21 J22] per element ---
J11 = sum(dNdk .* ptx, 2);
J12 = sum(dNdk .* pty, 2);
J21 = sum(dNde .* ptx, 2);
J22 = sum(dNde .* pty, 2);
Jdet_all = J11.*J22 - J12.*J21;

% --- InvJ * [dN/dksi; dN/deta] -> [dN/dx; dN/dy] per element ---
invDet = 1 ./ Jdet_all;
dNdx = invDet .* ( J22 .* dNdk - J12 .* dNde);
dNdy = invDet .* (-J21 .* dNdk + J11 .* dNde);

% --- Build N_all (2 x 16 x nEle) ---
Nvals = [N1, N2, N3, N4, N5, N6, N7, N8];
N_all = zeros(2, 16, nEle);
for k = 1:8
    N_all(1, 2*k-1, :) = Nvals(:, k);
    N_all(2, 2*k,   :) = Nvals(:, k);
end

% --- Build DN_all (4 x 16 x nEle) ---
DN_all = zeros(4, 16, nEle);
for k = 1:8
    DN_all(1, 2*k-1, :) = dNdx(:, k);
    DN_all(2, 2*k-1, :) = dNdy(:, k);
    DN_all(3, 2*k,   :) = dNdx(:, k);
    DN_all(4, 2*k,   :) = dNdy(:, k);
end

end
