# Frozen image-domain compensation and PSF-preservation diagnostic

Written before new image decoding/outcomes, 2026-09-28. Follow-up to regional
motion, not production integration. No detector, new fit, threshold tuning,
Jetson job, annotation use, RAW16, new download or sealed-holdout access.

## Fixed inputs and operator

Use only the eight prior pairs 0/1,42/43,85/86,127/128,170/171,212/213,255/256,
298/299, approved native 2448×2048 U8 PNGs, and regional_motion_01/result.json
SHA256 `3161a56132bd5de41273ed495b360050bec286e43108259785307be431d7f822`.
Approved download-validation SHA256 is
`b602b89755e60122f1cac200808d19bc58d55443697ff5d16f83da8afde530d9`.
Reuse the hash-pinned PNG loader from verify_aot_image_patches.py unchanged.
Whitelist exactly these16 images before decoding; verify PNG and decoded hashes.

All48 saved cell fits, training gates and coherent hulls remain unchanged.
No refitting to pixels or injected targets. Three saved arms: guarded global
translation, local translation, local affine. The guarded global reference is
also piecewise, because its fit differs per withheld cell; it is NOT one global
image warp. There is no blending, winner selection, mask filling or fallback.

Output coordinates p are previous-native pixel centers at integer coordinates.
For each output pixel use its own native6×8 cell's previous→current matrix F.
Pull current at q=F(p), NOT inverse(F)(p); subtract previous as signed float32.
Use CPU OpenCV INTER_CUBIC with float32 images/maps and constant-zero border
storage. Quantized remap coordinates come from convertMaps(CV_16SC2); require
the complete4×4 quantized source footprint inside the native image, including
zero-weight taps (conservative). Combine that with unchanged pointwise hull and
training eligibility; erode combined support by2 native output pixels using a
5×5 all-ones square and false border. Invalid output/residual pixels are NaN plus
an explicit false mask, never valid zero evidence. Record pre/post erosion masks.
No claim of byte-exact production CUDA/warpPerspective parity: this is a new
regional remap harness using the production interpolation family, not the
production composed-transform or learned temporal-background path.

## Dense clean image census

For each pair/arm, stream construction of full native maps/masks by bounded
stripes and warp each ORIGINAL current image once. Report native pixel support
counts before/after erosion, per48 cells and upper/lower halves. Also report
absolute signed-pair-residual median/p90/max/RMS on each arm's support and on
the identical pairwise and all-three common sets. These are image differences, not geometric
truth or target nuisance rates; exposure, moving content and correspondence
errors remain mixed. Store mask/field identities and cell-seam map-jump summaries.

## Real-background injections, fixed before pixels

Main sites: all48 native cell centers with (+0.25,+0.25)px added. At each site,
test circular Gaussian PSFs with sigma0.6/1.2px, signed peak ±16DN, and current
native independent offsets (0.5,0) or(2,-1)px. There are8 probes/site, 3072
pair/site/probe records, and9216 arm evaluations. No favorable-site replacement.

For all arms use IDENTICAL previous/current injected images. The previous PSF
center is p0. The current center is p0+d_global(p0)+delta, where d_global is the
saved guarded global translation for the anchor cell; it is a shared nominal
injection trajectory, not physical scene-motion truth. Gaussian amplitude is
peak DN, not integrated flux. Truncate at6sigma Euclidean radius. Injection is
clip(rint(U8+PSF),0,255), and measured effective increments use injected−original
U8 in float32. Record intended and realized signed peak/positive/negative mass,
clipped pixel count and quantization difference independently of warp effects.

For source-context efficiency, crop only the bounded native rectangle containing
all output ROI cubic footprints plus the injected-current PSF. Previous/output
ROIs are65×65 around floor(p0). Outside-native coordinates remain invalid.
Apply the unchanged full-image support masks, with sufficient context for their
erosion; never erode a tiny ROI independently or treat a crop edge as image edge.

Compute clean residual R and injected residual R' with the same frozen field
and support. D=R'−R isolates transfer; compare it separately to the continuous
PSF oracle T_current(F(p))−T_previous(p), evaluated analytically, not remapped.
Record actual current-only target increment as well, including peak, positive
and negative mass, L1/L2, signed-template response and positive-mass centroid
(polarity-normalized). Report oracle discrepancies, clipping and masking, not
an assumed100% preservation requirement. Near-zero oracle/responses yield null
ratios, not division by zero. Continuous oracles ignore U8 quantization by design;
quantization loss is reported separately. Outside a numerically available fit,
an oracle is undefined; no identity/global replacement is used.

Signed-template response is dot(observed,oracle)/sqrt(sum(oracle²)) in DN;
matched gain divides the same dot product by sum(oracle²). L2 denotes the
Euclidean norm, not its square. Denominators≤1e-12 produce null ratios. All
error comparisons use identical valid pixels. Oracle absolute-mass coverage is
conditional on numerical field definition, whose separate fraction is retained;
it must not hide pixels with no fit. Full-support eligibility requires full
numerical definition and the entire NONEMPTY significant target footprint
supported. That footprint is the UNION of abs(current PSF oracle) and
abs(previous PSF oracle)>1e-4 times the intended peak magnitude, preventing
signed cancellation from hiding missing target support. Report residual-oracle
footprint support separately. ROI partialness and complete65×65 ROI support are
separate fields, not synonyms for this target-footprint criterion. Full-support
eligibility additionally excludes native source-PSF truncation, an absent
significant current-target oracle, or significant current/previous PSF values on
the metric ROI boundary (possible metric truncation). These remain explicit
partial/truncated records, not successful full-preservation cases.

Visibility check is distinct: compare the isolated signed-template response
with clean residual RMS and the clean residual's response to the SAME fixed
oracle template on identical valid pixels. Retain both absolute DN and ratios;
zero clean noise yields null ratios. This is not a detector or calibrated SNR.
Do not call algebraic target cancellation against a paired background a visibility
pass. Report valid ROI fraction and oracle absolute-mass coverage. Partially
masked/truncated cases remain labeled partial, not full-preservation successes.

Boundary sweeps use four fixed anchors (1224,853.25),(1224,1194.25),
(1071.25,1024),(1377.25,1024), stepping −1,−0.5,0,0.5,1px normal to the boundary.
Use sigma0.6, ±16DN, delta(0.5,0). Nominal d_global is frozen at the anchor for
ALL five positions, preventing the tested piecewise field from defining the
trajectory. These320 probes/960 arm evaluations are a spatial-continuity test
under each fixed pair's field, not consecutive-frame tracking. Retain support
changes, centroid sequences and finite adjacent centroid jumps; missing values
remain missing. Never bridge a support gap with a zero or predicted observation.

Border probes: p0=(1.25,1.25),(2446.25,1.25),(1.25,2046.25),(2446.25,2046.25),
both widths/signs, delta(0.5,0):128 probes/384 arm evaluations. Keep all losses.
Total actual-background workload:3520 probes/10560 arm evaluations.

## Generated numerical and temporal controls

Generated arrays only may be tested before freeze; actual imagery may not.
Verify cubic pull against independent scalar quantized cubic (a=−0.75), including
identity, signed integer and fractional translations, affine/shear, ramp/impulse,
pixel borders, masked input and absence of wrapping. Numerical tolerance1e-4DN
for float32 interpolation,1e-8 for native coordinate algebra. Verify input
immutability, NaN/mask agreement, no unsupported fallback and exactly-once source
sampling. Include guards against manifest/file/output changes.

Generated nine-frame trajectories use native-coordinate analytic backgrounds
and independently rendered PSFs, never recursively warped prior images.
Reference target path p_t=(1222+0.5t,1024+0.0625t²), t=0..8. Camera fields:
identity; translation (0.25t,−0.125t); affine A_t=[[1+0.001t,0.002t],
[−0.001t,1−0.0005t]], about center(1224,1024), plus(0.25t,−0.125t).
Render current target center F_t(p_t), so reference target motion is known.
Use sigma0.6/1.2, amplitude±8/±32, initial phase(0,0),(0.25,0.25),(0.5,0.5):
72 trajectories ×9 frames. Background is128+12sin(x/17)+9cos(y/23), evaluated
at F_t inverse q before U8 quantization. Add one separate support-dropout version
of each camera's sigma0.6,+16,zero-phase trajectory: frames3/4 unavailable,
others available. No filling missing frames; recovery samples the original frame.
Store centroid/peak/energy error and missing counts, and adjacent residual-lobe
metrics versus independent analytic PSFs. Static zero-motion repeated-frame
controls must cancel; zero target residual there is expected, not erasure.
The static inventory is exactly eight nine-frame identity/static-target controls:
two widths × four signed amplitudes, zero phase, target fixed at(1222,1024), with
identical original frames. Thus there are83 generated trajectories /747 frames.

These controls do not test estimator contamination (fits are frozen), target
identity, detector recall or real-video temporal validity. Generated temporal
results must never be relabeled as consecutive observations from the eight
widely separated actual frame pairs.

## Freeze, audit and delivery

Bind plan, core, runner, generated-control code/tests, unchanged helper and input
hashes before any new real-image results. New exclusive output image_preservation_01.
Check source/code/manifest identities before/after; preserve failures/results.
Report all declared denominators and partial/unavailable outcomes. Numerical
assertions are conformance checks, not new image-quality acceptance thresholds.
No result-derived threshold tuning or automatic production promotion.

Fixed visual sample before outcomes: pairs0 and212, native cell IDs0,27,47,
sigma0.6,+16,delta(0.5,0), all three arms; plus the first vertical and horizontal
seam five-position sweeps for pair0. Include unsupported cases. Render native
grayscale crops, support overlays and signed residuals with a fixed ±16DN scale;
any enlargement uses nearest neighbor and is explicitly labeled.

Independent audit: pairs0 and298, cells0,27,47, sigma0.6,+16,delta(0.5,0), all
arms, retaining failures. Recompute bounded source-pixel cubic samples and metric
accounting from original approved images; audit all workload/cohort denominators.
Document results and compact artifacts locally. Proceed toward a detector rerun
only if image/temporal safety evidence supports it; do not silently integrate.
