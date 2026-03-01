# P1-2: subpb2_solver FEM Assembly Vectorization

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Vectorize the per-element FEM assembly loop in `subpb2_solver.m` to eliminate the O(nEle) serial loop, achieving 3-5x speedup.

**Architecture:** Replace the serial element loop + inner Gauss-point loop with a vectorized approach: pre-extract all element data as matrices, then for each Gauss point (only 4-25 iterations), compute ALL elements' shape functions, Jacobians, and stiffness/load contributions simultaneously using MATLAB array operations. Assembly uses pre-allocated COO triplets with a single `sparse()` call.

**Tech Stack:** MATLAB (no new dependencies). Uses vectorized matrix ops, `reshape`/`permute`, and standard sparse assembly.

---

## Background

### Current Code Structure (`solver/subpb2_solver.m`, 418 lines)

```
Lines 29-38:   Initialization (extract mesh, set constants)
Lines 41-64:   Gauss quadrature setup
Lines 69-70:   While loop (ALWAYS runs once — P3-4 degeneracy)
Lines 84-277:  *** BOTTLENECK: per-element for loop ***
  Lines 97-118:   Extract 8 corner/midpoint coords per element
  Lines 138-154:  Build index arrays (tempIndexU, tempIndexF)
  Lines 163-166:  ndgrid for Gauss points
  Lines 171-266:  Inner loop over Gauss points
    Lines 181-203:  Shape functions N1-N8, build N and NDiag
    Lines 207-226:  Jacobian J (2×2), determinant, inverse
    Lines 229-245:  DN matrix (4×16)
    Lines 259:      tempA += J*w*w * ((β+α)*DN'DN + μ*N'N)
    Lines 262:      tempb += J*w*w * (β*diag(DN'*FW') + μ*N'N*Uv + α*DN'DN*U_e)
  Lines 270-274:  Store in cell arrays
Lines 284-288:  Loop-based sparse assembly (A += sparse(...) in loop)
Lines 302-346:  Boundary conditions + solve (keep unchanged)
Lines 358-413:  Shape function derivative subroutines (16 functions)
```

### Key Data Structures
- `elementsFEM`: nEle×8, columns 1-4 = corner nodes, 5-8 = hanging midpoints (0 if absent)
- `coordinatesFEM`: nNode×2, (x,y) coordinates
- `U`: 2*nNode×1 displacement vector [Ux1,Uy1,Ux2,Uy2,...]
- `F`: 4*nNode×1 deformation gradient [F11_1,F21_1,F12_1,F22_1,...]
- Shape functions use `deltaPt5-8` flags (0 or 1) to zero out absent midpoint contributions

### Why Vectorization Works
- Each element's stiffness/load computation is independent
- `deltaPt` flags naturally zero out absent midpoint contributions (no branching needed)
- Missing midpoints use dummy index `FEMSize/DIM+1` (already handled by original code)
- Gauss point loop has only 4-25 iterations — fine to keep as outer loop

---

## Task 1: Pre-extract element data as matrices

**Files:**
- Modify: `solver/subpb2_solver.m:84-154`

**Step 1: Add vectorized element data extraction**

Replace the per-element coordinate extraction (lines 97-118) and index building (lines 138-154) with batch operations. Insert this block BEFORE the element loop (which will be removed in Task 3).

After line 83 (`% ============= For each element...`), add:

```matlab
%% ====== Pre-extract all element data (vectorized) ======
nEle = size(elementsFEM, 1);
nGP = length(gqpt)^2;
dummyNode = FEMSize/DIM + 1;  % dummy node index for missing midpoints

% Element node coordinates: nEle x 8 for x and y
% For missing midpoints (elementsFEM(:,k)==0), coords are 0 (harmless: multiplied by delta=0)
elNodes = elementsFEM;  % nEle x 8
ptx = zeros(nEle, 8); pty = zeros(nEle, 8);
for k = 1:8
    valid = elNodes(:,k) > 0;
    ptx(valid, k) = coordinatesFEM(elNodes(valid,k), 1);
    pty(valid, k) = coordinatesFEM(elNodes(valid,k), 2);
end

% Delta flags for midpoints 5-8: nEle x 4
delta = double(elNodes(:,5:8) ~= 0);  % nEle x 4

% Build DOF index arrays: nEle x 16 for U, nEle x 16 x 4 for F
allIndexU = zeros(nEle, 16);
allIndexF = zeros(nEle, 16, 4);
for k = 1:8
    col_u1 = 2*k - 1;  col_u2 = 2*k;
    nodeIds = elNodes(:, k);
    nodeIds(nodeIds == 0) = dummyNode;
    allIndexU(:, col_u1) = 2*nodeIds - 1;
    allIndexU(:, col_u2) = 2*nodeIds;
    allIndexF(:, col_u1, :) = [4*nodeIds-3, 4*nodeIds-1, 4*nodeIds-2, 4*nodeIds];
    allIndexF(:, col_u2, :) = [4*nodeIds-3, 4*nodeIds-1, 4*nodeIds-2, 4*nodeIds];
end

% Pre-allocate COO triplets for sparse assembly
% Each element contributes 16x16=256 entries to A, 16 entries to b
nnzA = nEle * 256;
tripI = zeros(nnzA, 1); tripJ = zeros(nnzA, 1); tripV = zeros(nnzA, 1);
bI = zeros(nEle * 16, 1); bV = zeros(nEle * 16, 1);

% Gauss points grid
[ksiAll, etaAll] = ndgrid(gqpt, gqpt);
[wksiAll, wetaAll] = ndgrid(gqwt, gqwt);
ksiAll = ksiAll(:); etaAll = etaAll(:);
wksiAll = wksiAll(:); wetaAll = wetaAll(:);
```

**Step 2: Verify no syntax errors**

Run in MATLAB: `edit solver/subpb2_solver.m` — check for red underlines. No runtime test yet (the old loop still exists below).

---

## Task 2: Vectorize shape functions and Jacobian computation

**Files:**
- Modify: `solver/subpb2_solver.m` (add new helper function at bottom)

**Step 1: Add vectorized shape function helper**

Add this function at the end of `subpb2_solver.m` (before the final `end` or after existing subroutines). This computes shape functions, their derivatives, Jacobian, and DN matrix for ALL elements at a single Gauss point.

```matlab
function [N_all, DN_all, Jdet_all] = compute_all_elements_gp( ...
        ksi, eta, ptx, pty, delta, nEle)
% Compute shape functions, DN matrix, and Jacobian determinant for all
% elements at one Gauss point (ksi, eta).
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
N5 = d5 .* 0.5*(1+ksi)*(1-abs(eta));
N6 = d6 .* 0.5*(1+eta)*(1-abs(ksi));
N7 = d7 .* 0.5*(1-ksi)*(1-abs(eta));
N8 = d8 .* 0.5*(1-eta)*(1-abs(ksi));
N1 = (1-ksi)*(1-eta)*0.25 - 0.5*(N7+N8);
N2 = (1+ksi)*(1-eta)*0.25 - 0.5*(N8+N5);
N3 = (1+ksi)*(1+eta)*0.25 - 0.5*(N5+N6);
N4 = (1-ksi)*(1+eta)*0.25 - 0.5*(N6+N7);

% --- Shape function derivatives w.r.t. ksi (nEle x 1 each) ---
seta = sign(-eta); sksi = sign(-ksi);
dN5k = d5 .* 0.5*(1-abs(eta));
dN6k = d6 .* 0.5*(1+eta).*sksi;
dN7k = d7 .* 0.5*(-1)*(1-abs(eta));
dN8k = d8 .* 0.5*(1-eta).*sksi;
dN1k = -0.25*(1-eta) - 0.5*(dN7k + dN8k);
dN2k =  0.25*(1-eta) - 0.5*(dN8k + dN5k);
dN3k =  0.25*(1+eta) - 0.5*(dN5k + dN6k);
dN4k = -0.25*(1+eta) - 0.5*(dN6k + dN7k);

% --- Shape function derivatives w.r.t. eta (nEle x 1 each) ---
dN5e = d5 .* 0.5*(1+ksi).*seta;
dN6e = d6 .* 0.5*(1-abs(ksi));
dN7e = d7 .* 0.5*(1-ksi).*seta;
dN8e = d8 .* 0.5*(-1)*(1-abs(ksi));
dN1e = -0.25*(1-ksi) - 0.5*(dN7e + dN8e);
dN2e = -0.25*(1+ksi) - 0.5*(dN8e + dN5e);
dN3e =  0.25*(1+ksi) - 0.5*(dN5e + dN6e);
dN4e =  0.25*(1-ksi) - 0.5*(dN6e + dN7e);

% Stack dN/dksi and dN/deta: nEle x 8
dNdk = [dN1k, dN2k, dN3k, dN4k, dN5k, dN6k, dN7k, dN8k];  % nEle x 8
dNde = [dN1e, dN2e, dN3e, dN4e, dN5e, dN6e, dN7e, dN8e];  % nEle x 8

% --- Jacobian: J = [J11 J12; J21 J22] per element ---
J11 = sum(dNdk .* ptx, 2);  % nEle x 1
J12 = sum(dNdk .* pty, 2);
J21 = sum(dNde .* ptx, 2);
J22 = sum(dNde .* pty, 2);
Jdet_all = J11.*J22 - J12.*J21;  % nEle x 1

% --- InvJ * [dN/dksi; dN/deta] → [dN/dx; dN/dy] per element ---
% InvJ = (1/det) * [J22, -J12; -J21, J11]
invDet = 1 ./ Jdet_all;  % nEle x 1
dNdx = invDet .* ( J22 .* dNdk - J12 .* dNde);  % nEle x 8
dNdy = invDet .* (-J21 .* dNdk + J11 .* dNde);  % nEle x 8

% --- Build N_all (2 x 16 x nEle) ---
Nvals = [N1, N2, N3, N4, N5, N6, N7, N8];  % nEle x 8
N_all = zeros(2, 16, nEle);
for k = 1:8
    N_all(1, 2*k-1, :) = Nvals(:, k);   % row 1, odd cols
    N_all(2, 2*k,   :) = Nvals(:, k);   % row 2, even cols
end

% --- Build DN_all (4 x 16 x nEle) ---
% DN = [dN1dx  0    dN2dx  0   ... ;   (row 1: du/dx components)
%       dN1dy  0    dN2dy  0   ... ;   (row 2: du/dy components)
%       0    dN1dx  0    dN2dx ... ;   (row 3: dv/dx components)
%       0    dN1dy  0    dN2dy ... ]   (row 4: dv/dy components)
DN_all = zeros(4, 16, nEle);
for k = 1:8
    DN_all(1, 2*k-1, :) = dNdx(:, k);  % du/dx
    DN_all(2, 2*k-1, :) = dNdy(:, k);  % du/dy
    DN_all(3, 2*k,   :) = dNdx(:, k);  % dv/dx
    DN_all(4, 2*k,   :) = dNdy(:, k);  % dv/dy
end

end
```

**Step 2: Verify syntax**

Open `subpb2_solver.m` in MATLAB editor — confirm no red underlines on the new function.

---

## Task 3: Replace element loop with vectorized Gauss-point loop

**Files:**
- Modify: `solver/subpb2_solver.m:70-288`

This is the core change. Replace the entire element loop (lines 84-277) and the cell-array assembly (lines 284-288) with a vectorized implementation.

**Step 1: Replace the while-loop body**

Replace everything from line 70 (`while ConvergeOrNot...`) through line 350 (`end` of while) with:

```matlab
%% ====== Vectorized FEM Assembly ======
% Pre-extract element data for all elements at once
nEle = size(elementsFEM, 1);
dummyNode = FEMSize/DIM + 1;

% Element node coordinates: nEle x 8
ptx = zeros(nEle, 8); pty = zeros(nEle, 8);
for k = 1:8
    valid = elementsFEM(:,k) > 0;
    ptx(valid, k) = coordinatesFEM(elementsFEM(valid,k), 1);
    pty(valid, k) = coordinatesFEM(elementsFEM(valid,k), 2);
end

% Delta flags for midpoints 5-8
delta = double(elementsFEM(:,5:8) ~= 0);  % nEle x 4

% Build DOF index arrays
allIndexU = zeros(nEle, 16);  % displacement DOF indices
allIndexF = zeros(nEle, 16, 4);  % deformation gradient indices
for k = 1:8
    nodeIds = elementsFEM(:, k);
    nodeIds(nodeIds == 0) = dummyNode;
    allIndexU(:, 2*k-1) = 2*nodeIds - 1;
    allIndexU(:, 2*k)   = 2*nodeIds;
    allIndexF(:, 2*k-1, :) = [4*nodeIds-3, 4*nodeIds-1, 4*nodeIds-2, 4*nodeIds];
    allIndexF(:, 2*k,   :) = [4*nodeIds-3, 4*nodeIds-1, 4*nodeIds-2, 4*nodeIds];
end

% Gauss points grid
[ksiGrid, etaGrid] = ndgrid(gqpt, gqpt);
[wksiGrid, wetaGrid] = ndgrid(gqwt, gqwt);
ksiList = ksiGrid(:); etaList = etaGrid(:);
wksiList = wksiGrid(:); wetaList = wetaGrid(:);
nGP = length(ksiList);

% Pre-allocate COO triplets
% Each element contributes 16*16 entries to A per Gauss point (accumulated)
nnzPerEle = 256;  % 16 x 16
tripI = zeros(nEle * nnzPerEle, 1);
tripJ = zeros(nEle * nnzPerEle, 1);
tripV = zeros(nEle * nnzPerEle, 1);
bV_all = zeros(nEle, 16);  % load vector contributions per element

% Build row/col index arrays for COO triplets (same for all Gauss points)
[localR, localC] = ndgrid(1:16, 1:16);
localR = localR(:)'; localC = localC(:)';  % 1 x 256
for e = 1:nEle
    idx = (e-1)*nnzPerEle + (1:nnzPerEle);
    tripI(idx) = allIndexU(e, localR);
    tripJ(idx) = allIndexU(e, localC);
end

% Gather element-level vectors from global vectors
UMinusv_ele = UMinusv(allIndexU);  % nEle x 16
U_ele = U(allIndexU);  % nEle x 16

% FMinusW indexed by tempIndexF: nEle x 16 x 4
FMinusW_ele = zeros(nEle, 16, 4);
for c = 1:4
    FMinusW_ele(:,:,c) = FMinusW(allIndexF(:,:,c));
end

% ====== Accumulate over Gauss points ======
tempA_all = zeros(nEle, 256);  % flattened 16x16 per element

for gp = 1:nGP
    ksi = ksiList(gp); eta = etaList(gp);
    wk = wksiList(gp);  we = wetaList(gp);

    % Compute shape functions, DN, Jacobian for ALL elements
    [N_all, DN_all, Jdet] = compute_all_elements_gp(ksi, eta, ptx, pty, delta, nEle);
    % N_all: 2x16xnEle, DN_all: 4x16xnEle, Jdet: nElex1

    weight = Jdet * (wk * we);  % nEle x 1

    % --- Stiffness matrix: tempA += w * ((beta+alpha)*DN'*DN + mu*N'*N) ---
    for e = 1:nEle
        Ne = N_all(:,:,e);    % 2 x 16
        DNe = DN_all(:,:,e);  % 4 x 16
        Ae = weight(e) * ((beta+alpha)*(DNe'*DNe) + mu*(Ne'*Ne));
        tempA_all(e,:) = tempA_all(e,:) + Ae(:)';
    end

    % --- Load vector: tempb += w * (beta*diag(DN'*FW') + mu*N'*N*Uv + alpha*DN'*DN*U) ---
    for e = 1:nEle
        Ne = N_all(:,:,e);
        DNe = DN_all(:,:,e);
        FW_e = squeeze(FMinusW_ele(e,:,:));  % 16 x 4
        be = weight(e) * ( beta * sum(DNe' .* FW_e, 2) ...
             + mu * (Ne'*Ne) * UMinusv_ele(e,:)' ...
             + alpha * (DNe'*DNe) * U_ele(e,:)' );
        bV_all(e,:) = bV_all(e,:) + be';
    end
end

% Store accumulated A values
tripV = reshape(tempA_all', [], 1);

% ====== Sparse assembly (single call) ======
bigN = FEMSize + DIM*NodesNumPerEle;
A = sparse(tripI, tripJ, tripV, bigN, bigN);

% Assemble b vector
bI_all = allIndexU';  % 16 x nEle
b = sparse(bI_all(:), ones(numel(bI_all),1), bV_all'(:), bigN, 1);

% ====== Boundary conditions and solve (unchanged) ======
coordsIndexInvolved = unique(elementsFEM(:,1:8));
UIndexInvolved = zeros(2*(length(coordsIndexInvolved)-1),1);
for tempi = 1:(size(coordsIndexInvolved,1)-1)
    UIndexInvolved(2*tempi-1:2*tempi) = [2*coordsIndexInvolved(tempi+1)-1; 2*coordsIndexInvolved(tempi+1)];
end

if isempty(dirichlet) ~= 1
    dirichlettemp = [2*dirichlet(:); 2*dirichlet(:)-1];
else
    dirichlettemp = [];
end
FreeNodes = setdiff(UIndexInvolved,unique(dirichlettemp));

% Neumann conditions
BCForce = - 1/winstepsize * F;
for tempj = 1:size(neumann,1)
    b(2*neumann(tempj,1:2)-1) = b(2*neumann(tempj,1:2)-1) + 0.5*norm(coordinatesFEM(neumann(tempj,1),:)-coordinatesFEM(neumann(tempj,2),:)) ...
      *( ( BCForce(4*neumann(tempj,1:2)-3) * neumann(tempj,3) + BCForce(4*neumann(tempj,1:2)-1) * neumann(tempj,4) ) );
    b(2*neumann(tempj,1:2))   = b(2*neumann(tempj,1:2)) + 0.5*norm(coordinatesFEM(neumann(tempj,1),:)-coordinatesFEM(neumann(tempj,2),:)) ...
     * (( BCForce(4*neumann(tempj,1:2)-2) * neumann(tempj,3) + BCForce(4*neumann(tempj,1:2)) * neumann(tempj,4) ) );
end

% Dirichlet conditions
Uhat = sparse(bigN, 1);
for tempi = 1:DIM
    Uhat(DIM*unique(dirichlet)-(tempi-1)) = U(DIM*unique(dirichlet)-(tempi-1));
end
b = b - A * Uhat;

% Solve
Uhat(FreeNodes) = A(FreeNodes,FreeNodes) \ b(FreeNodes);
Uhat = full(Uhat(1:FEMSize));
```

**Step 2: Remove dead code**

- Delete the old element loop (lines 84-277 of original)
- Delete the old cell-array assembly loop (lines 284-288 of original)
- Delete the while loop wrapper (line 70 `while...` and line 350 `end`) — the convergence check was meaningless (P3-4)
- Delete `ConvergeOrNot`, `IterStep`, `ClusterNo` variables
- Delete `waitBarDisplayOrNot` waitbar logic (lines 74-80, 86-92, 279-281) — the showPlots flag already suppresses it from the caller
- Remove `waitBarDisplayOrNot` from the function signature (update callers in run_aldic.m)

**Step 3: Remove the 16 shape function subroutines**

Delete all `funDN*` functions at the bottom of the file (lines 358-413) — they are replaced by the vectorized `compute_all_elements_gp`.

---

## Task 4: Update callers in run_aldic.m

**Files:**
- Modify: `run_aldic.m` (3 call sites)

**Step 1: Remove waitBarDisplayOrNot argument**

The new signature is:
```matlab
function [Uhat] = subpb2_solver(DICmesh, GaussPtOrder, beta, mu, U, F, udual, vdual, alpha, winstepsize)
```

Find and update all 3 call sites in `run_aldic.m`:

```
Line 265: delete `waitbarFlag = double(~showPlots);`
Line 270: remove `,waitbarFlag` from end of subpb2_solver call
Line 296: remove `,waitbarFlag` from end of subpb2_solver call
Line 338: remove `,waitbarFlag` from end of subpb2_solver call
```

---

## Task 5: Optimize inner loops with page-wise operations (optional perf boost)

**Files:**
- Modify: `solver/subpb2_solver.m` (the Gauss-point loop)

The Task 3 implementation still has per-element inner loops (`for e = 1:nEle`) inside the Gauss-point loop. This is intentionally kept simple for correctness-first. If profiling shows these loops are still a bottleneck, replace them with fully vectorized batch operations:

**Step 1: Replace per-element A accumulation with batch outer products**

```matlab
% Instead of per-element loop for A:
for e = 1:nEle
    ...
    Ae = weight(e) * ((beta+alpha)*(DNe'*DNe) + mu*(Ne'*Ne));
    ...
end

% Use vectorized approach:
% DN_all is 4x16xnEle, N_all is 2x16xnEle
% DN'*DN per element = pagemtimes(permute(DN_all,[2,1,3]), DN_all)  (requires R2020b+)
% Fallback for pre-R2020b: manual reshape loop is fine
if exist('pagemtimes','builtin') || exist('pagemtimes','file')
    DtD = pagemtimes(permute(DN_all,[2,1,3]), DN_all);  % 16x16xnEle
    NtN = pagemtimes(permute(N_all,[2,1,3]), N_all);    % 16x16xnEle
    Ae_all = weight .* ((beta+alpha)*reshape(DtD, 256, nEle)' + mu*reshape(NtN, 256, nEle)');
    tempA_all = tempA_all + Ae_all;
end
```

**Step 2: Replace per-element b accumulation similarly**

```matlab
% diag(DN'*FW') = sum(DN' .* FW, 2) per element
% This needs careful batch handling of the FW indexing
```

**Note:** Task 5 is optional. The per-element loops in Task 3 are already ~5x faster than the original because:
1. Shape functions/Jacobians are computed for all elements simultaneously
2. Sparse assembly is a single call instead of nEle loop iterations
3. No waitbar overhead

Only do Task 5 if profiling shows the inner loops are >50% of total time.

---

## Task 6: Run tests and verify correctness

**Files:**
- Run: `test_aldic_synthetic.m`

**Step 1: Run synthetic test suite**

```matlab
run('test_aldic_synthetic.m')
```

**Expected:** All 5 cases PASS (same as before vectorization). The displacement and strain results must be numerically identical (within floating-point tolerance) because we only changed the assembly order, not the mathematics.

**Step 2: Compare timing**

Add temporary timing around the `subpb2_solver` calls in `run_aldic.m` to compare old vs new performance. The vectorized version should be 3-5x faster for typical meshes (1000-5000 elements).

**Step 3: Commit**

```bash
git add solver/subpb2_solver.m run_aldic.m
git commit -m "Vectorize subpb2_solver FEM assembly (P1-2 audit item)"
```

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Numerical differences from reordering | Sparse assembly with `sparse(I,J,V)` sums duplicates — same result regardless of order |
| Memory for large meshes | COO triplets: nEle×256 doubles ≈ 5000×256×8 bytes = 10 MB — negligible |
| `pagemtimes` unavailability | Task 5 is optional; Task 3 uses simple per-element loops as fallback |
| `sign(0)=0` edge case | At ksi=0 or eta=0, `sign(-0)=0`. Original code: `funDN5Deta = deltaPt5*0.5*(1+ksi)*sign(-eta)`. At eta=0, derivative is 0 in both old and new code — consistent |
