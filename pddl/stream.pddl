(define (stream nvidia-tamp)
  (:rule
    :inputs (?o ?g ?gty ?o2)
    :domain (and (IsGraspType ?o ?g ?gty) (AdmitsGraspType ?o2 ?gty))
    :certified (AdmitsGrasp ?o ?g ?o2))
  (:rule
    :inputs (?o1 ?wp1 ?rp ?o2 ?wp2)
    :domain (and (PoseKin ?o1 ?wp1 ?rp ?o2 ?wp2) (Value ?rp))
    :certified (Value ?wp1))
  (:rule
    :inputs (?o1 ?wp1 ?rp ?o2 ?wp2)
    :domain (and (PoseKin ?o1 ?wp1 ?rp ?o2 ?wp2) (Sample ?rp))
    :certified (Sample ?wp1))
  (:rule
    :inputs (?o1 ?wp1 ?rp ?o2 ?wp2)
    :domain (and (PoseKin ?o1 ?wp1 ?rp ?o2 ?wp2) (Dist ?rp))
    :certified (Dist ?wp1))
  (:rule
    :inputs (?o1 ?wp1 ?rp ?o2 ?wp2 ?bq)
    :domain (and (PoseKin ?o1 ?wp1 ?rp ?o2 ?wp2) (NearPose ?o2 ?wp2 ?bq) (Posterior ?rp))
    :certified (NearPose ?o1 ?wp1 ?bq))

  ;(:stream test-not-equal
  ;  :inputs (?o1 ?2)
  ;  :domain (and (Object ?o1) (Object ?o2))
  ;  :certified (NEq ?o1 ?o2))

  (:stream sample-grasp
    :inputs (?o ?gty)
    :domain (ValidGraspType ?o ?gty)
    :outputs (?g)
    :certified (and (Grasp ?o ?g) (IsGraspType ?o ?g ?gty))) ; TODO: produce carry conf for ?o


  (:stream sample-pose
    :inputs (?o ?r)
    :domain (and (MovableBase) (Stackable ?o ?r))
    :outputs (?rp)
    :certified (and (RelPose ?o ?rp ?r)
                    (Value ?rp) (Sample ?rp) (BeliefUpdate ?o ?rp @none ?rp)
                ))
  (:stream sample-nearby-pose
    :inputs (?o1 ?o2 ?wp2 ?bq)
    :domain (and (NearPose ?o2 ?wp2 ?bq) (Stackable ?o1 ?o2)) ; TODO: ensure door is open?
    :outputs (?wp1 ?rp)
    :certified (and (RelPose ?o1 ?rp ?o2) (NearPose ?o1 ?wp1 ?bq)
                    (Value ?wp1) (Sample ?wp1)
                    (Value ?rp) (Sample ?rp) (BeliefUpdate ?o1 ?rp @none ?rp)
                    (WorldPose ?o1 ?wp1) (PoseKin ?o1 ?wp1 ?rp ?o2 ?wp2)))

  ; Movable base
  (:stream plan-pick
    :inputs (?o ?wp ?g) ; ?aq0)
    :domain (and (MovableBase) (WorldPose ?o ?wp) (Sample ?wp) (Grasp ?o ?g)) ; (RestAConf ?aq0))
    :outputs (?bq ?aq ?at)
    :certified (and (BConf ?bq) (ATraj ?at)
                    ; (AConf ?bq ?aq0)
                    (AConf ?bq @rest_aq) ; (AConf ?bq @calibrate_aq)
                    (AConf ?bq ?aq)
                    (Pick ?o ?wp ?g ?bq ?aq ?at)))
  (:stream plan-pull
    :inputs (?j ?a1 ?a2) ; ?aq0)
    :domain (and (MovableBase) (Angle ?j ?a1) (Angle ?j ?a2)) ; (RestAConf ?aq0))
    :outputs (?bq ?aq1 ?aq2 ?at)
    :certified (and (BConf ?bq) (ATraj ?at)
                    ; (AConf ?bq ?aq0)
                    ; TODO: strange effect with plan constraints
                    (AConf ?bq @rest_aq) ; (AConf ?bq @calibrate_aq)
                    (AConf ?bq ?aq1) (AConf ?bq ?aq2)
                    (Pull ?j ?a1 ?a2 ?bq ?aq1 ?aq2 ?at)))

  (:stream plan-press
    :inputs (?k)
    :domain (and (MovableBase) (Knob ?k)) ; TODO: (WorldPose ?o ?wp)
    :outputs (?bq ?aq ?at)
    :certified (and (BConf ?bq) (ATraj ?at)
                    ; (AConf ?bq ?aq0)
                    (AConf ?bq @rest_aq) ; (AConf ?bq @calibrate_aq)
                    (AConf ?bq ?aq)
                    (Press ?k ?bq ?aq ?at)))
  (:stream fixed-plan-press
    :inputs (?k ?bq)
    :domain (NearJoint ?k ?bq) ; TODO: use pose instead?
    :outputs (?aq ?at)
    :certified (and (ATraj ?at) (AConf ?bq ?aq)
                    (Press ?k ?bq ?aq ?at)))

	(:stream fixed-plan-pour
		:inputs (?bowl ?wp ?cup ?g ?bq) ; TODO: can pour ?bowl & ?cup
		:domain (and (Bowl ?bowl) (WorldPose ?bowl ?wp) (Sample ?wp) (NearPose ?bowl ?wp ?bq)
                 (Pourable ?cup) (Grasp ?cup ?g))
		:outputs (?aq ?at)
		:certified (and (Pour ?bowl ?wp ?cup ?g ?bq ?aq ?at)
						        (AConf ?bq ?aq) (ATraj ?at)))

  ; Fixed base
  (:stream test-near-pose
    :inputs (?o ?wp ?bq)
    :domain (and (WorldPose ?o ?wp) (CheckNearby ?o) (Sample ?wp) (InitBConf ?bq))
    :certified (NearPose ?o ?wp ?bq))
  (:stream fixed-plan-pick
    :inputs (?o ?wp ?g ?bq)
    :domain (and (NearPose ?o ?wp ?bq) ; (InitBConf ?bq)
                 (WorldPose ?o ?wp) (Sample ?wp) (Grasp ?o ?g))
    :outputs (?aq ?at)
    :certified (and (ATraj ?at) (AConf ?bq ?aq)
                    (Pick ?o ?wp ?g ?bq ?aq ?at)))

  (:stream test-near-joint
    :inputs (?j ?bq)
    :domain (and (Joint ?j) (InitBConf ?bq))
    :certified (NearJoint ?j ?bq))
  (:stream fixed-plan-pull
    :inputs (?j ?a1 ?a2 ?bq)
    :domain (and (Angle ?j ?a1) (Angle ?j ?a2) (NearJoint ?j ?bq))
    :outputs (?aq1 ?aq2 ?at)
    :certified (and (ATraj ?at) (AConf ?bq ?aq1) (AConf ?bq ?aq2)
                    (Pull ?j ?a1 ?a2 ?bq ?aq1 ?aq2 ?at)))

  (:stream plan-base-motion
    :fluents (AtGConf AtWorldPose AtGrasp) ; AtBConf, AtAConf, AtGConf, AtAngle
    :inputs (?bq1 ?bq2 ?aq) ; TODO: just rest_aq?
    :domain (and (MovableBase) (AConf ?bq1 ?aq) (AConf ?bq2 ?aq))
    :outputs (?bt)
    :certified (BaseMotion ?bq1 ?bq2 ?aq ?bt))
  (:stream plan-arm-motion
    :fluents (AtGConf AtWorldPose AtGrasp) ; AtBConf, AtAConf, AtGConf, AtAngle
    ; TODO: disjunctive stream conditions
    :inputs (?bq ?aq1 ?aq2)
    :domain (and (AConf ?bq ?aq1) (AConf ?bq ?aq2))
    :outputs (?at)
    :certified (ArmMotion ?bq ?aq1 ?aq2 ?at))
  (:stream plan-gripper-motion
    :inputs (?gq1 ?gq2)
    :domain (and (GConf ?gq1) (GConf ?gq2))
    :outputs (?gt)
    :certified (GripperMotion ?gq1 ?gq2 ?gt))
  ;(:stream plan-calibrate-motion
  ;  :inputs (?bq) ; ?aq0)
  ;  :domain (and (BConf ?bq)) ; (RestAConf ?aq0))
  ;  :outputs (?at) ; ?aq
  ;  :certified (and (ATraj ?at)
  ;                  ;(CalibrateMotion ?bq ?aq0 ?at))
  ;                  (CalibrateMotion ?bq @rest_aq ?at))
  ;)

  (:stream sample-observation ; TODO: sample-nearby-observation?
    :inputs (?o1 ?rp1 ?o2)
    :domain (and (RelPose ?o1 ?rp1 ?o2) (Dist ?rp1))
    :outputs (?obs)
    :certified (Observation ?o1 ?o2 ?obs))
  (:stream update-belief
    :inputs (?o1 ?rp1 ?o2 ?obs)
    :domain (and (RelPose ?o1 ?rp1 ?o2) (Dist ?rp1) (Observation ?o1 ?o2 ?obs))
    :outputs (?rp2) ; TODO: could add parameter ?o2 as well
    :certified (and (RelPose ?o1 ?rp2 ?o2) (BeliefUpdate ?o1 ?rp1 ?obs ?rp2)
                    (Posterior ?rp2) (Sample ?rp2)))
  (:stream compute-detect
    :inputs (?o ?wp)
    :domain (and (WorldPose ?o ?wp) (Sample ?wp))
    :outputs (?r)
    :certified (and (Ray ?r) (Detect ?o ?wp ?r)))

  (:stream compute-pose-kin
    :inputs (?o1 ?rp ?o2 ?wp2)
    :domain (and (RelPose ?o1 ?rp ?o2) (WorldPose ?o2 ?wp2))
    :outputs (?wp1)
    :certified (and (WorldPose ?o1 ?wp1) (PoseKin ?o1 ?wp1 ?rp ?o2 ?wp2))
    ; (PoseTriplet ?o1 ?wp1 ?rp) ; For instantiation?
  )
  ;(:stream compute-angle-kin
  ;  :inputs (?o ?j ?a)
  ;  :domain (and (Connected ?o ?j) (Angle ?j ?a))
  ;  :outputs (?wp)
  ;  :certified (and (WorldPose ?o ?wp) (AngleKin ?o ?wp ?j ?a))
  ;)

  (:stream test-cfree-worldpose
    :inputs (?o1 ?wp1)
    :domain (WorldPose ?o1 ?wp1)
    :certified (CFreeWorldPose ?o1 ?wp1))
  (:stream test-cfree-worldpose-worldpose
    :inputs (?o1 ?wp1 ?o2 ?wp2)
    :domain (and (WorldPose ?o1 ?wp1) (WorldPose ?o2 ?wp2))
    :certified (CFreeWorldPoseWorldPose ?o1 ?wp1 ?o2 ?wp2))
  (:stream test-cfree-pose-pose
    :inputs (?o1 ?rp1 ?o2 ?rp2 ?s)
    :domain (and (RelPose ?o1 ?rp1 ?s) (RelPose ?o2 ?rp2 ?s))
    :certified (CFreeRelPoseRelPose ?o1 ?rp1 ?o2 ?rp2 ?s))
  (:stream test-cfree-approach-pose
    :inputs (?o1 ?wp1 ?g1 ?o2 ?wp2)
    :domain (and (WorldPose ?o1 ?wp1) (Grasp ?o1 ?g1) (WorldPose ?o2 ?wp2))
    :certified (CFreeApproachPose ?o1 ?wp1 ?g1 ?o2 ?wp2))
  (:stream test-cfree-angle-angle
    :inputs (?j1 ?a1 ?a2 ?o2 ?wp2)
    :domain (and (Angle ?j1 ?a1) (Angle ?j1 ?a2) (WorldPose ?o2 ?wp2))
    :certified (CFreeAngleTraj ?j1 ?a1 ?a2 ?o2 ?wp2))
  (:stream test-cfree-bconf-pose
    :inputs (?bq ?o2 ?wp2)
    :domain (and (BConf ?bq) (WorldPose ?o2 ?wp2))
    :certified (CFreeBConfPose ?bq ?o2 ?wp2))
  (:stream test-cfree-traj-pose
    :inputs (?at ?o2 ?wp2)
    :domain (and (ATraj ?at) (WorldPose ?o2 ?wp2))
    :certified (CFreeTrajPose ?at ?o2 ?wp2))

  (:stream test-ofree-ray-pose
    :inputs (?r ?o ?wp)
    :domain (and (Ray ?r) (WorldPose ?o ?wp))
    :certified (OFreeRayPose ?r ?o ?wp))
  (:stream test-ofree-ray-grasp
    :inputs (?r ?bq ?aq ?o ?g)
    :domain (and (Ray ?r) (AConf ?bq ?aq) (Grasp ?o ?g))
    :certified (OFreeRayGrasp ?r ?bq ?aq ?o ?g))

  ; TODO: these could also just be populated in the initial state
  ;(:stream test-gripper
  ;  :inputs (?gq)
  ;  :domain (GConf ?gq)
  ;  :certified (OpenGConf ?gq))
  (:stream test-door
    :inputs (?j ?a ?s)
    :domain (and (Angle ?j ?a) (Status ?s))
    :certified (AngleWithin ?j ?a ?s))

  (:function (DetectCost ?o ?rp1 ?obs ?rp2) ; TODO: pass in risk level
             (BeliefUpdate ?o ?rp1 ?obs ?rp2))
  ;(:function (Distance ?bq1 ?bq2)
  ;  (and (BConf ?bq1) (BConf ?bq2))
  ;)
  ;(:function (MoveCost ?t)
  ;  (and (BTraj ?t))
  ;)
)