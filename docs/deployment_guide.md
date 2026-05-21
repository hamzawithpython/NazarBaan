# NazarBaan — Gate Deployment Guide

This document specifies the hardware, camera placement, and operational
configuration required to deploy NazarBaan at a real Pakistani housing-society
gate. It complements the technical performance reports under `reports/` by
addressing the *real-world* factors that determine whether the trained model
actually works in production.

## 1. Camera selection

| Parameter | Recommendation | Why |
|---|---|---|
| Resolution | 4 MP minimum (2688×1520) | Pakistani plates are ~30-50 cm wide; at typical 5 m gate distance the plate fills ~5-8% of the frame; below 4 MP the characters become too small for OCR |
| Shutter | Global shutter preferred; rolling shutter ≤ 1/500 s | Slow shutter blurs plate text on moving cars; rolling-shutter cameras must compensate with very fast exposure |
| Frame rate | 15-25 fps | More than enough for gate traffic; 30+ fps wastes CPU |
| Lens | 4-8 mm varifocal, IP66+ housing | Varifocal lets the installer adjust framing on-site; IP66 protects against monsoon |
| Night capability | Built-in IR illumination, IR-cut filter | Headlights at night blow out untreated plates; IR illumination preserves contrast |

## 2. Camera placement

| Parameter | Recommendation |
|---|---|
| Height | 2.5-3.5 m above ground |
| Distance from boom barrier | 5-8 m back |
| Downward angle | 20-30° (not steeper — distorts plate aspect ratio) |
| Horizontal alignment | Centered on the lane; not at an oblique angle |
| Focus | Manually locked at the boom distance, not autofocus |

A common mistake is mounting the camera on the boom-barrier housing itself.
This puts the camera too close, at too steep an angle, and at too low a height
— all three are detection killers. The camera belongs on a separate mast or
on the gate-house wall, set back from the boom.

## 3. Trigger zone configuration

The pipeline uses a fixed rectangular trigger zone (a region in the camera
frame) to define "the gate." A vehicle's plate must enter this zone for the
event to be logged.

**The zone is configured once, at install time**, by:

1. Capturing a still frame of the camera feed with a test vehicle stopped at
   the boom barrier.
2. Drawing the trigger zone tightly around the plate position in that frame.
3. Saving the zone coordinates to the deployment config.

The zone does not change after that — the camera is fixed, the boom is fixed.

The synthetic test video used during development does not satisfy this
condition (each frame is a different photograph taken from a different
angle), so some plates appear outside the development trigger zone. This is
an artifact of the test data, not a property of real gate footage.

## 4. Lighting

| Condition | Recommendation |
|---|---|
| Daytime | Avoid direct backlight (camera pointed at the sun); orient the camera so daylight falls on the plate, not behind it |
| Night | IR illumination, not visible-light floodlights (which annoy drivers) |
| Rain | Hood the camera lens; angle the housing so water beads run off rather than pool |

## 5. Compute hardware

The full pipeline (YOLOv8n detection at imgsz=960 + EasyOCR + tracking)
runs at:

- 3.15 FPS sustained on a 2-core/4-thread 6th-gen Intel laptop CPU.
- ~6-8 FPS estimated on an Intel N100 mini-PC (~$150).
- ~15-25 FPS estimated on an Intel N305 or modern i3 mini-PC ($300-500).

For typical gate traffic (one car every 5-30 seconds), even the lowest tier is
sufficient. The recommended deployment unit is an N100 fanless mini-PC mounted
inside the gate-house, connected to the camera over Ethernet (RTSP stream).

## 6. Operator UX

The system's OCR layer achieves 44% exact-match accuracy on held-out test
plates — strong as a starting suggestion, weak as an unsupervised log.
Every commercial ANPR system in production handles this the same way: the
gate operator sees the OCR reading and confirms or corrects it before the
boom barrier opens. The Phase 8 Streamlit app implements exactly this
workflow.

Every correction becomes training data. After approximately 500 corrections
per gate, the dataset is rich enough to fine-tune the OCR recognition model
specifically for that gate's plates, lighting, and angle. This is the
strongest long-term improvement path.

## 7. Known limitations

- **Severely blurred or angled plates** (motion blur from fast-moving cars,
  plates seen at >45° azimuth) will not be read. Mitigation: camera
  placement per Section 2.
- **Non-standard plates** (handwritten, custom-painted, Urdu-only) are
  out of scope for this model. Mitigation: most provincial registries
  enforce printed plates; non-compliant vehicles are rare and can be
  handled by manual entry.
- **Two-vehicle simultaneous arrival** at the same trigger zone will assign
  one track but log only one plate. Mitigation: use two cameras (one per
  lane) for multi-lane gates.