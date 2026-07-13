# Engineering Struggles and How They Were Resolved

Companion to the software setup document. Each entry is a real problem hit
during development, its root cause, and the resolution. These map directly to
"design iterations", "lessons learned", and "limitations" material for the
paper.

## 1. Hardware and electrical

### Burned power wiring from a stalled servo
The servo power rail was first wired with thin breadboard jumper wire. During
an actuation test the firmware travel limits had been removed to gain range;
a servo was commanded past the mechanism's end stop, could not reach its
setpoint, and drew sustained stall current. The undersized conductors
overheated to failure. Root cause was the combination of unconstrained travel
and wiring not rated for stall current. Resolution: the firmware travel
limits were restored permanently and the extra range was obtained
mechanically through the linkage instead. Lesson: never widen actuator limits
to compensate for a mechanical shortcoming, and treat stall current as the
design current for power wiring.

### First linkage had too little travel
The first coupling screwed a circular servo horn directly to the board. The
servo's usable swing through the firmware's safe pulse range produced too
little board rotation. Resolution: a push-rod linkage (straight horn, ball
joint rod ends on a threaded rod, wooden lever on the board) whose lever
ratios provide the needed range while the firmware limits stay conservative.

### Static friction dominates at small tilts
Below a measurable tilt the ball simply does not move. Early axis
calibration pulses at low amplitude produced zero displacement and useless
measurements. The calibration tool had to escalate pulse amplitude per axis
until the ball demonstrably moved, and the controller later needed explicit
stiction compensation (see controls section).

### Asymmetric and slack linkages
One axis initially moved the ball far less than the other (loose horn screw,
slack in rod ends), producing degenerate axis-map measurements. The servo
channels were also physically swapped relative to the firmware's assumption
at one point. Resolution: the axis mapping is measured, not assumed - a
calibration script pulses each servo axis and records the ball's response,
absorbing any swaps, sign flips, or asymmetries into a measured matrix.

## 2. Calibration

### Wrong board dimensions propagated silently
Three different play-area sizes circulated at different times (a mistyped
3220 x 2820, an assumed 322 x 282, and the final measured 263 x 222 mm).
Because the homography scales everything to the configured dimensions, every
derived artifact was silently wrong until re-measured with a ruler. Lesson:
physical dimensions are measurements, not config defaults, and every
downstream artifact must be regenerated when they change.

### ChArUco calibration attempted and abandoned
A ChArUco marker board was tried as an automatic calibration target, glued
to a side platform next to the board. A homography is only valid for points
in a single plane, and the pattern's plane was not the plane the ball rolls
on, so the calibration was systematically warped. After the geometric
problem was understood, the approach was abandoned entirely rather than
reworked: the manual corner-click calibration was already accurate to a few
millimetres, takes under a minute, and became the standard method. Lesson:
an automated calibration is only worth its complexity if it is geometrically
valid, and a simple manual procedure that is demonstrably sufficient beats a
sophisticated one that is not.

### The camera is not a constant
The OS camera index changed between reboots and USB ports, so scripts
sometimes silently opened the laptop webcam or a virtual camera instead of
the maze camera. Auto-exposure made the first frames of every capture too
dark for brightness-based seeding. Resolutions: a per-machine config overlay
for the device index, and never seeding from the first frames of a stream.
The capture backend and frame rate turned out to be a much larger problem in
their own right, described next.

### The camera was silently running at a tenth of its frame rate
The camera is rated 120 fps at 1280x800 but was delivering about ten.
Profiling ruled out the tracker, the overlay drawing, and the serial write
(all together under ten milliseconds); the missing eighty-odd milliseconds per
iteration was the frame read itself, blocking while it waited for the camera.
On Windows the default DirectShow backend exposed only the camera's
uncompressed mode, whose bandwidth over USB 2 caps 1280x800 at about ten
frames per second, and this particular camera would not negotiate a compressed
mode at all. An earlier belief that selecting a compressed format had fixed the
frame rate was simply wrong; the format was never actually accepted. The fix
was to switch to the Media Foundation backend, which selects the camera's
native high-rate mode and delivers the full 120 fps at full resolution with no
loss of resolution, at the cost of a slow one-time open. Lesson: a driver
reporting a frame rate is not the same as delivering it, so measure the achieved
rate; and the capture backend is a first-class design parameter, not a detail.

### Opening the camera once and sharing it
Because the high-rate backend is slow to open (tens of seconds), paying that
cost in every calibration and test script made iteration painful. A small
camera-server process now opens the device once and publishes each frame into
shared memory; every other tool attaches to it in milliseconds and falls back
to opening the device directly when the server is not running. As a side
benefit the server timestamps each frame at capture time, which removed a class
of phantom velocity spikes described in the perception section.

## 3. Perception

### The ball looks like the holes
At the camera's resolution a reflective silver marble on a bright board is
nearly indistinguishable from the dark holes and their bright rims. A naive
bright-blob detector locked onto wall glints and hole rims; automatic
seeding regularly picked glare instead of the ball. The working detector
combines two cues that holes cannot satisfy simultaneously: motion
(frame-to-frame difference; holes do not move) and specular highlight (the
metal glint saturates brighter than printed features), plus an offline
calibration that blacklists locations that are bright too often across a
recording (hole rims, fixed glare) and a region-of-interest polygon that
excludes everything off the playable surface. For demos, seeding is manual:
the operator clicks the ball. Lesson: on this kind of scene, tracking
reliability came from cue combination and precomputed scene knowledge, not
from tuning a single threshold.

### Stationary balls disappear
Motion-based detection produces no signal when the ball stops, which is
exactly when the controller most needs feedback (stall detection). Gaps are
bridged by the highlight cue and short constant-velocity prediction, and
lighting must keep the glint above the specular threshold, which varied
between lab sessions and needed re-measurement.

### Phantom velocities from burst-delivered frames
Once the camera ran at its true rate, it occasionally delivered two frames a
fraction of a millisecond apart. The velocity estimate, a finite difference of
position over time, then divided a sub-millimetre detection jitter by a
near-zero time step and reported impossible speeds of several hundred
millimetres per second. Those spikes tripped the safety logic (emergency
braking and the composure state) continually, which held the ball in place and
prevented any progress - and looked, misleadingly, like a control failure.
Resolution: the estimator ignores the velocity contribution of frames closer
together than a minimum time step and clamps any single measured velocity above
a physical ceiling before it enters the smoothing filter. Lesson: a
finite-difference derivative is only as trustworthy as its timestamps.

## 4. Planning and control

### The "unstabilizable" ball was a ten-hertz control loop
For a long stretch the controller could not hold the ball steady: it rang and
oscillated around every target, and effort went into control gains, braking
authority, and a stabilization state, none of which helped much. The run logs
eventually showed the real cause: the control loop was executing at about ten
iterations per second, not the camera's rated 120. At that rate the ball moves
several millimetres between updates and the velocity estimate lags a full
frame, so any controller over-corrects and rings; the instability was a
sampling-rate problem, not a gains problem, and the root cause was the camera
frame-rate issue described in the calibration section. Lesson: before tuning a
controller, confirm the loop is actually running at the rate it assumes, because
an undersampled loop cannot be stabilized by any choice of gains.

### The wall mask parked the ball at the start
A freshly regenerated wall mask silently stopped the ball from ever leaving the
start. The runtime wall-proximity speed limit, intended as protection for when
the ball drifts off the route, was also being applied while the ball was on the
route, where it double-counted the wall clearance already folded into the
precomputed speed plan. With a dense mask it throttled the on-route target speed
to about a third, the controller judged the barely-creeping ball to be near its
target, and it never commanded enough tilt to actually move. Resolution: the
runtime wall speed limit now applies only once the ball has drifted off the
centerline; on the route the precomputed plan alone governs speed. Lesson: two
mechanisms that each cap the same quantity must not silently compose.

### Composure: stopping instead of chasing after a disturbance
When a hard brake or a disturbance briefly made the ball fast and unstable, the
follower kept demanding forward progress, which fed the oscillation and drove
the ball into holes. A composure state was added: on a genuine runaway the
controller stops pursuing progress, holds position and damps the ball, and
resumes only once it is back under control. Tuning it was itself a struggle -
an early version triggered on ordinary overshoot and on the phantom velocity
spikes described earlier, so it was active for most of a run and froze the ball
in place; it was then narrowed to fire only on a real runaway (speed far above
the plan) and to release as soon as the ball settles rather than after a fixed
hold. Even narrowed, it was ultimately switched off in the final build: once
speed control became smooth enough that genuine runaways stopped happening (see
"From a pile of band-aids to speed control" below), a state whose whole job was
to catch a runaway had nothing left to catch, and holding position on a false
trigger did more harm than the disturbance it was guarding against.

### An oblique view of the walls fires false wall contacts
The overhead camera views the walls at an angle, so a ball near a wall can
appear to be touching it when it is not. A local replanner that rerouted the
ball whenever its line of sight to the next waypoint looked wall-blocked
therefore triggered constantly on false contacts and hijacked the intended
path. It was disabled by default; ordinary path following, the off-route wall
slowdown, and the emergency brake cover recovery without it. Lesson: a
perception input that is known to be geometrically unreliable should not be
allowed to gate an aggressive automatic action.

### Backlash in one axis masqueraded as a weak axis
Returning to the asymmetric-linkage problem with better instrumentation: one
pitch direction consistently measured about half the ball displacement of the
others during axis calibration, and the calibration faithfully recorded that,
producing a map that boosted the axis to compensate. On the rig the axis then
over-drove and oscillated. Driving the servo directly, bypassing the
controller, revealed the true cause: the pitch linkage had backlash, so small
commands were absorbed by mechanical slop and produced no motion at all, and
only a large command suddenly broke through and lurched the ball. A calibration
that faithfully measures a mechanically broken axis produces a faithfully wrong
map, and no software correction can recover the lost authority. The software
change is only diagnostic - the calibration tool now warns when one direction
moves far less than the others, which almost always means the ball hit a wall
during that pulse or the linkage has slop - and the mechanical slack must be
removed. Lesson: distrust an asymmetric calibration result and inspect the
mechanism before trusting the numbers.

### Pure position control parks the ball short
With proportional-derivative position control, the commanded tilt shrinks as
the ball approaches its target and falls below the tilt needed to overcome
static friction: the system reaches a stable equilibrium with the ball
parked short of the target ("balanced but not moving", observed for seconds
at a time in run logs). Resolution: explicit stiction compensation (a
minimum command magnitude once a commanded-but-not-moving state persists)
plus a small integral term.

### The stiction fix then caused corner crashes
The first stiction kick triggered on any single slow frame, but the ball is
also intentionally slow while braking into corners; the kick punched full
breakaway tilt into deliberate slowdowns and flung the ball into walls and
holes. Resolution: the kick requires the low-speed condition to persist
(distinguishing real static friction from intentional braking), and all
commands pass through a slew-rate limiter before reaching the hardware.

### From a pile of band-aids to speed control
Even with the persistence gate, the discrete stall kick was only the first of
several reactive patches, each fixing a symptom of the same disease. The kick
freed a dead stall but not a ball twitching in place at a tight corner (the
velocity estimate read the twitch as "moving" and reset the kick), so a
separate displacement-based "unstick" push was added to detect stuck by actual
net travel and shove the ball toward the target. The unstick and the kick could
both over-push near a hole, so both had to be capped and suppressed inside
capture zones. The composure state (above) was a third patch, catching the
runaways the first two occasionally caused. The result was a stack of discrete
mechanisms that each guarded against the others' failure modes and interacted
in ways that were hard to reason about.

The root cause underneath all of them was that the controller was fighting the
plant instead of modelling it: board tilt sets the ball's ACCELERATION, and the
ball is a static-friction plant, so any controller that commands zero tilt at
zero error lets stiction pin the ball short of the target -- and then needs a
kick to escape. The redesign reframed the whole loop around controlling speed
directly: a velocity PI whose integral term IS the stiction compensator. The
integral winds up smoothly while the ball is slower than planned (including a
dead stall) until the tilt breaks it free, then unwinds as the speed error
closes, with an anti-windup bound so a long stall can never wind up into a
launch. Continuous speed control made the ball trackable everywhere, and the
three discrete band-aids -- stall kick, unstick, composure -- became
unnecessary and were disabled. They remain in the codebase, behind config
flags, in case a future rig with worse stiction needs them again. Lesson: a
stack of reactive patches that each catch another patch's failure is a sign the
plant is being fought rather than modelled; fixing the model retired all of
them at once.

### The last few holes needed the wall, not more control
With smooth speed control the ball completed almost the whole maze, but a
handful of holes sat so close to the traced route that ordinary wobble still
occasionally dropped the ball in -- notably one hole at the foot of a downhill
straight (the ball arrived fast and got flung sideways into it) and two holes
right at the finish (where the ball arrives hottest). Tightening gains or
adding more braking did not help, because the route itself passed within a
ball's wobble of the hole. The fix was geometric rather than control-theoretic:
a wall-hug detour that shifts the lookahead point a few millimetres
perpendicular to the path through those spots, so the ball rides along the
outer wall around the hole and rejoins the route afterward, plus a gentle
one-sided "wall-lean" push at the very last hole that leans the ball into the
wall (and fades out once it is already leaning, so it never oscillates back
toward the hole). Combined with the finish-approach crawl, this closed the last
few percent of reliability. Lesson: when the route grazes a hazard, moving the
target away from the hazard beats asking the controller to track a bad target
more precisely.

### Reactive last-resort brakes, retired for the same reason
A last-resort emergency brake -- discard path following and slam a full brake
opposite the velocity when the ball's trajectory is about to carry it into a
hole -- was built as a final safety net. Like the composure state, it earned
its retirement: tilt is force, so a hard brake still applied as the ball
reaches zero speed launches it backward, and on a near-miss the violent
reaction repeatedly did more damage than the near-miss would have. Once speed
planning and the wall-hug detours kept the ball from arriving at a hole out of
control in the first place, the emergency brake was disabled in the final build
(it too remains behind a config flag). The final safety posture is prevention
by smooth planning, not violent last-second reaction. Lesson: a reactive safety
net that acts through the same nonlinear actuator it is trying to tame can be
more dangerous than the event it guards against; prevent the condition upstream
instead.

### Safety caps can silently disable safety fixes
A run with the command cap set below the stiction kick clipped the kick away
entirely: the ball received a command it physically could not respond to,
for the whole run, with no error anywhere. A loud warning now fires when the
configuration is self-defeating. Lesson: interacting safety mechanisms need
explicit consistency checks.

### The maze's geometry defeats naive path following
The route snakes, so corridors that are far apart along the route sit
millimetres apart behind a single wall. Projecting the ball onto the nearest
route segment sometimes associated it with a corridor on the other side of a
wall, and the controller then drove the ball into that wall. The first fix
(only allowing the association to move within a window of path progress per
frame) still failed inside chicanes, where the adjacent corridor is within
the window. The robust fix required wall knowledge: the walls are rasterized
once into a static obstacle mask, candidate associations whose line of sight
from the ball crosses a wall are rejected, and wall proximity imposes a
speed limit.

### Chicanes read as straight lines
Corner slowdown initially measured path curvature as the angle between the
tangent here and the tangent a fixed distance ahead. A chicane turns right
then left, the two turns cancel at the endpoints, and the measure reported
"straight" - so the ball entered chicanes at full cruise speed and
ricocheted between the walls. Resolution: curvature is accumulated absolute
turning along the span, which cannot cancel.

## 5. Process and collaboration

### Two operating systems, one repo
The team develops on macOS and Windows simultaneously. Serial ports
(/dev/cu.* vs COM), camera indices, python vs python3, and shell syntax
(bash line continuations vs PowerShell) all differ, and shared config edits
kept overwriting each machine's settings. Resolution: a gitignored
per-machine configuration overlay for device-specific values, and
documentation written for both shells.

### Parallel work on shared artifacts
Calibration artifacts and annotation CSVs were regenerated independently on
different machines, causing merge conflicts and stale-artifact confusion
(one machine's homography with another machine's path file silently
misaligns everything). Resolution: an explicit dependency rule - a new
homography invalidates and requires regenerating every derived artifact -
and backups of replaced artifacts committed alongside regenerated ones.

### Debugging by measurement, not impression
Early tuning was driven by watching the ball and guessing. Progress
accelerated after every run began writing a full log and an analysis tool
reported detection rate, progress reached, cross-track error statistics, and
stall episodes located by path position. Several supposed "controller
problems" turned out to be a too-low command cap, a wrong corridor
association, or a tracking dropout - all visible in the logs and invisible
to the eye. This pattern recurred repeatedly and became the strongest lesson
of the project: an apparent control failure was, in turn, a ten-hertz control
loop caused by a mis-negotiated camera mode, phantom velocities caused by
near-zero frame time steps, an invisible speed wall caused by a wall mask
applied on the route, and a wiggling axis caused by mechanical backlash. In
each case the fix followed directly from a measurement (loop rate, frame time
steps, the runtime speed scale, direct servo-response amplitudes) and would
have been nearly impossible to reach by watching the ball. The recurring trap
was attributing an infrastructure or hardware fault to the controller; the
recurring remedy was to instrument the suspected layer and read the number.
