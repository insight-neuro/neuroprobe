# Changelog

## 0.1.8

### Added
- Direct MNI152 / MNI305 coordinates for all 10 Braintreebank subjects, bundled in `neuroprobe/mni_coords/`.
- `BrainTreebankSubject(..., coordinates_type=...)` now accepts `"mni"` (= `"mni152"`), `"mni152"`, and `"mni305"` in addition to `"cortical"` and `"lpi"`.
- `get_electrode_coordinates_mni(version="mni152")` — replaces the previous `NotImplementedError` stub.

### Fixed
- README logo now uses `raw.githubusercontent.com` URL so it renders on PyPI.

## 0.1.7 and earlier

(historical — see git log)
