# Paste-ready corrections for the Final Report Draft

Verified against the code as it stands on `main` today. Each block says what to
replace. Only the software sections change; hardware/mechanical/CV text is left
alone.

---

## FIX 1 — Section 3 (System Overview), the loop-cadence sentence

**Replace:**
> One iteration is executed every other camera frame: the system detects the
> ball, converts the detected position into board coordinates, finds where the
> ball is along the annotated route, computes a tilt command, and sends the
> command to the Arduino.

**With:**
> Each iteration acts on the most recently captured frame from a single-frame
> buffer — so the loop always uses the newest image, never a stale queued one —
> and detects the ball, converts the detected position into board coordinates,
> finds where the ball is along the annotated route, computes a tilt command,
> and sends the command to the Arduino. The loop runs as fast as detection,
> overlay drawing, and logging allow, below the camera's 120 fps capture rate,
> so it consumes a subset of captured frames rather than a fixed one-in-N.

---

## FIX 2 — Section 10 (Path Planning), the A* recovery paragraph

**Replace:**
> An automatic recovery mechanism using A* pathfinding activates when the ball
> is significantly off-route, stuck near a wall, or stalled for a sustained
> duration. A short recovery path is planned around the nearest obstacles back
> toward the intended route, using the existing wall map for obstacle
> avoidance. A recovery candidate is only accepted if it does not increase the
> ball's distance from the intended route.

**With:**
> An automatic A* recovery mechanism is implemented but disabled in the final
> build. When enabled it plans a short path around the nearest obstacles back
> toward the route when the ball drifts significantly off-route or stalls, using
> the wall map for obstacle avoidance, and accepts a recovery candidate only if
> it does not increase the ball's distance from the route. In practice the
> overhead camera's oblique view of the walls produced false wall contacts that
> made the replanner fire spuriously and hijack the intended path, so it was
> switched off; ordinary path following plus the off-route wall slowdown cover
> recovery without it. It is retained behind a configuration flag (see
> Section 15.x, Retired approaches).

Also, in the same section, the sentence that currently reads
"…back toward the intended route" is fine; only the framing above changes.

---

## FIX 3 — Section 11 (Control Algorithm)

Keep the **Axis mapping**, **Calibration/identity-matrix**, and **Neutral trim**
paragraphs as they are — they are accurate. Replace everything from
"Controller and Tuned Gains" onward with the following.

### Controller and tuned gains

The default and final controller is **carrot following with velocity feedback**:
it chases the furthest lookahead ("carrot") point on the route still in
unobstructed line of sight, and it controls the ball's *velocity* toward that
point rather than its position. Board tilt commands acceleration, so a velocity
loop brakes automatically when the ball carries too much speed into a corner —
something a position loop cannot do.

The velocity loop is a **PI controller on speed error**. With
*e = v_desired − v_measured* (where *v_desired* is the planned speed aimed at the
carrot):

    board_cmd = k_vel · e  +  clamp( k_vel_i · ∫ e dt ,  ± I_max )

The proportional term tracks the planned speed. The **integral term is the
explicit stiction compensator**: the ball is a static-friction plant, so a pure
proportional command falls to zero exactly when the speed error does, and static
friction then pins the ball short of its target. The integral instead winds up
smoothly while the ball is slower than planned — including a dead stall — until
the tilt breaks it free, then unwinds as the error closes. An anti-windup bound
(*I_max*) caps the integral's contribution so a long stall can never wind up into
a launch. This continuous compensator replaced an earlier stack of discrete
"band-aids" (a breakaway stall kick, a displacement-based unstick push, and a
composure hold-and-damp state), discussed in Section 15.x.

**Final values:** k_vel = 0.035, k_vel_i = 0.05, integral cap I_max = 0.7,
driving command cap = 0.9, brake cap = 0.9, driving slew = 8 /s, brake slew =
12 /s, cruise speed v_max = 25 mm/s. The legacy discrete stall kick is set to 0
(disabled — replaced by the integral term).

Two other controllers exist in the code but were not used for final testing:
- **Velocity following** — tracks the path's local tangent directly.
- **Position following (legacy)** — a lookahead PD approach using full PID gains
  (kp, kd, ki); this is the only mode that uses the kp/kd/ki gains described in
  earlier drafts, and it was not the final controller.

### Command authority and safety

Command authority is **asymmetric**. Driving tilt is held to a gentle cap and
slew rate for smooth motion, while a command that opposes the ball's motion (a
brake) may use more authority and reverse faster. Because tilt is a force, a
brake tilt still applied as the ball reaches zero speed would launch it
backward, so the brake ceiling is speed-proportional and melts to nearly flat at
the stop. All commands are clamped and slew-limited before reaching the
hardware, and a warning fires if the command cap is set below a configured stall
kick (a self-defeating configuration).

### Speed planning

A single **speed plan for the whole route** is computed once at startup and
looked up slightly ahead of the ball, so slowdowns are anticipated rather than
discovered late. It folds together, as one speed-by-position profile:
corner curvature, committed hole-pass speeds, planned wall clearance, per-hole
"danger" crawls (holes the route passes close to at a sharp turn or nearly
grazes on a straight), and a **finish-approach crawl** for the last stretch,
which threads several holes just before the goal. A backward pass then
guarantees every slowdown is reachable by braking and a forward pass smooths
acceleration. A separate runtime wall slowdown applies only once the ball drifts
off the centerline, so it does not double-count clearance already in the plan.

### Wall-hug detours and end-game aids

For the few holes the traced route grazes, the controller's lookahead point is
shifted a few millimetres perpendicular to the path — a **wall-hug detour** — so
the ball rides along the outer wall around the hole and rejoins the route after
it, instead of oscillating into the capture zone. This is a planning-time offset
on the pursued point, not a change to the stored route. At the very last hole a
gentle one-sided **wall-lean** push leans the ball against the outer wall and
fades out once the ball is already moving into it (so it never oscillates back
toward the hole). Once the goal is reached, a damped **finish hold** keeps the
ball settled in the goal pocket until the operator ends the run, rather than
snapping straight to neutral.

---

## FIX 4 — Section 12 (Safety and Reliability), the "Emergency braking" bullet

**Replace:**
> Emergency braking and recovery. If the ball is about to enter a hole beyond
> normal control, path-following is discarded for a full brake opposite its
> velocity, bypassing the rate limiter. After this or any detected runaway, the
> system holds the ball with a small damping controller until it settles, then
> resumes normal control with a full reset.

**With:**
> Prevention over reaction. A last-resort emergency brake (discard path
> following and slam a full brake opposite the velocity when the trajectory is
> about to enter a hole) and a composure state (hold and damp the ball after a
> runaway) are both implemented but disabled in the final build. Because tilt is
> a force, a hard brake still applied as the ball stops launches it backward,
> and on a near-miss these violent reactions repeatedly did more damage than the
> near-miss itself. With smooth speed planning and the wall-hug detours keeping
> the ball from arriving at a hole out of control in the first place, both were
> switched off; they remain behind configuration flags (see Section 15.x). A
> slow ball drifting within a few millimetres of a wall still receives a small
> corrective push away from it.

The other Section 12 bullets (Arming, Ball-loss/neutral-on-loss, Stale
wall-mask protection, Guaranteed neutral on exit) are accurate — keep them.

> ⚠️ Also check **Table 5 (Section 14, Results)**: it currently lists
> "emergency braking … observed during runs" as a safety mechanism that
> triggered correctly. Emergency braking is disabled in the final build, so
> either reword this to the mechanisms that are actually active (neutral-on-loss,
> watchdog return, stale-mask disable, wall-escape push) or note that emergency
> braking was observed only in earlier development runs.

---

## ADD — new subsection "15.x Retired approaches" (put at the end of Section 15, Discussion)

**Retired approaches (design iterations).** Several control mechanisms were built,
used during development, and ultimately disabled once a cleaner solution made
them unnecessary. They remain in the codebase behind configuration flags, both
as a record of the design path and in case a future rig with worse mechanics
needs them.

- **Discrete stall kick.** A breakaway command floor that punched a fixed tilt
  into the ball once it had been commanded-but-not-moving for a sustained time,
  to overcome static friction. It worked but was crude: it had to be gated to
  avoid firing during intentional corner braking, and near a hole a ramped kick
  could fling the ball in. Replaced by the velocity-PI integral term, which
  compensates stiction continuously instead of in discrete punches.

- **Displacement-based unstick.** The stall kick could not free a ball twitching
  in place at a tight corner (its velocity estimate read the twitch as "moving"
  and reset the kick), so a second mechanism detected "stuck" by actual net
  displacement over a window and added a damped, capped push toward the target.
  It too became unnecessary once speed control was smooth, and it was disabled.

- **Composure / stabilize state.** After a hard brake or disturbance the ball
  could briefly go fast and unstable while the follower kept demanding progress,
  feeding an oscillation. A composure state stopped pursuing progress, held
  position, and damped the ball until it settled. It was hard to tune (early
  versions fired on ordinary overshoot and on phantom velocity spikes, freezing
  the ball for most of a run), and once genuine runaways stopped happening it had
  nothing left to catch, so it was disabled.

- **Emergency hole-brake.** A last-resort full brake opposite the velocity when
  the trajectory was about to enter a hole. Because tilt is a force, braking a
  ball to a stop launched it backward, and the reaction often did more harm than
  the near-miss; disabled once prevention (speed planning + detours) made it
  redundant.

- **A\* recovery replanner.** An A* search that rerouted the ball around
  obstacles back to the route when it drifted far off-route. The overhead
  camera's oblique view of the walls produced false wall contacts that made it
  fire spuriously and hijack the path, so it was disabled by default.

The through-line: each of these was a reactive patch for a symptom of one root
cause — a controller that commanded zero tilt at zero error and so let static
friction pin the ball. Modelling the ball as a static-friction plant and
controlling its speed with a PI loop (whose integral is the stiction
compensator) removed the cause, and the patches retired themselves.
