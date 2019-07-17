(define (stream nvidia-tamp)
  (:rule
    :inputs (?o ?g ?gty ?o2)
    :domain (and (IsGraspType ?o ?g ?gty) (AdmitsGraspType ?o2 ?gty))
    :certified (AdmitsGrasp ?o ?g ?o2)
  )
  (:stream sample-grasp
    :inputs (?o ?gty)
    :domain (and (Graspable ?o) (GraspType ?gty))
    :outputs (?g)
    :certified (and (Grasp ?o ?g) (IsGraspType ?o ?g ?gty)) ; TODO: produce carry conf
  )

  (:stream sample-pose
    :inputs (?o ?r)
    :domain (Stackable ?o ?r)
    :outputs (?rp)
    :certified (RelPose ?o ?rp ?r)
  )
  ;(:stream sample-nearby-pose
  ;  :inputs (?o1 ?o2 ?p2 ?bq)
  ;  :domain (and (Stackable ?o1 ?o2) (WorldPose ?o2 ?p2) (InitBConf ?bq))
  ;  :outputs (?p1 ?rp)
  ;  :certified (and (RelPose ?o1 ?rp ?o2) (NearPose ?o1 ?p1 ?bq)
  ;                  (WorldPose ?o1 ?p1) (PoseKin ?o1 ?p1 ?rp ?o2 ?p2))
  ;)

  ; Movable base
  (:stream plan-pick
    :inputs (?o ?p ?g)
    :domain (and (WorldPose ?o ?p) (Grasp ?o ?g) (MovableBase))
    :outputs (?bq ?aq ?at)
    :certified (and (BConf ?bq) (ATraj ?at)
                    (AConf ?bq @rest_aq) (AConf ?bq ?aq)
                    (Pick ?o ?p ?g ?bq ?aq ?at))
  )
  (:stream plan-pull
    :inputs (?j ?a1 ?a2)
    :domain (and (Angle ?j ?a1) (Angle ?j ?a2) (MovableBase))
    :outputs (?bq ?aq1 ?aq2 ?at)
    :certified (and (BConf ?bq) (ATraj ?at)
                    (AConf ?bq @rest_aq) (AConf ?bq ?aq1) (AConf ?bq ?aq2)
                    (Pull ?j ?a1 ?a2 ?bq ?aq1 ?aq2 ?at))
  )

  ; Fixed base
  (:stream test-near-pose
    :inputs (?o ?p ?bq)
    :domain (and (WorldPose ?o ?p) (Graspable ?o) (InitBConf ?bq))
    :certified (NearPose ?o ?p ?bq)
  )
  (:stream fixed-plan-pick
    :inputs (?o ?p ?g ?bq)
    :domain (and (WorldPose ?o ?p) (Grasp ?o ?g) (NearPose ?o ?p ?bq))
    :outputs (?aq ?at)
    :certified (and (ATraj ?at) (AConf ?bq ?aq)
                    (Pick ?o ?p ?g ?bq ?aq ?at))
  )
  (:stream test-near-joint
    :inputs (?j ?bq)
    :domain (and (Joint ?j) (InitBConf ?bq))
    :certified (NearJoint ?j ?bq)
  )
  (:stream fixed-plan-pull
    :inputs (?j ?a1 ?a2 ?bq)
    :domain (and (Angle ?j ?a1) (Angle ?j ?a2) (NearJoint ?j ?bq))
    :outputs (?aq1 ?aq2 ?at)
    :certified (and (ATraj ?at) (AConf ?bq ?aq1) (AConf ?bq ?aq2)
                    (Pull ?j ?a1 ?a2 ?bq ?aq1 ?aq2 ?at))
  )

  (:stream plan-base-motion
    :fluents (AtGConf AtWorldPose AtGrasp) ; AtBConf, AtAConf, AtGConf, AtAngle
    :inputs (?bq1 ?bq2 ?aq)
    :domain (and (AConf ?bq1 ?aq) (AConf ?bq2 ?aq) (MovableBase))
    :outputs (?bt)
    :certified (and ; (BTraj ?bt)
                    (BaseMotion ?bq1 ?bq2 ?aq ?bt))
  )
  (:stream plan-arm-motion
    :fluents (AtGConf AtWorldPose AtGrasp) ; AtBConf, AtAConf, AtGConf, AtAngle
    :inputs (?bq ?aq1 ?aq2)
    :domain (and (AConf ?bq ?aq1) (AConf ?bq ?aq2))
    :outputs (?at)
    :certified (and ; (ATraj ?bt)
                    (ArmMotion ?bq ?aq1 ?aq2 ?at))
  )
  (:stream plan-gripper-motion
    :inputs (?gq1 ?gq2)
    :domain (and (GConf ?gq1) (GConf ?gq2))
    :outputs (?gt)
    :certified (GripperMotion ?gq1 ?gq2 ?gt)
  )
  (:stream plan-calibrate-motion
    :inputs (?bq)
    :domain (BConf ?bq)
    :outputs (?at) ; ?aq
    :certified (and (ATraj ?at) (CalibrateMotion ?bq @rest_aq ?at))
  )

  (:stream compute-pose-kin
    :inputs (?o1 ?rp ?o2 ?p2)
    :domain (and (RelPose ?o1 ?rp ?o2) (WorldPose ?o2 ?p2))
    :outputs (?p1)
    :certified (and (WorldPose ?o1 ?p1) (PoseKin ?o1 ?p1 ?rp ?o2 ?p2))
    ; (PoseTriplet ?o1 ?p1 ?rp) ; For instantiation?
  )
  ;(:stream compute-angle-kin
  ;  :inputs (?o ?j ?a)
  ;  :domain (and (Connected ?o ?j) (Angle ?j ?a))
  ;  :outputs (?p)
  ;  :certified (and (WorldPose ?o ?p) (AngleKin ?o ?p ?j ?a))
  ;)

  (:stream test-cfree-pose-pose
    :inputs (?o1 ?rp1 ?o2 ?rp2 ?s)
    :domain (and (RelPose ?o1 ?rp1 ?s) (RelPose ?o2 ?rp2 ?s))
    :certified (CFreeRelPoseRelPose ?o1 ?rp1 ?o2 ?rp2 ?s)
  )
  (:stream test-cfree-approach-pose
    :inputs (?o1 ?p1 ?g1 ?o2 ?p2)
    :domain (and (WorldPose ?o1 ?p1) (Grasp ?o1 ?g1) (WorldPose ?o2 ?p2))
    :certified (CFreeApproachPose ?o1 ?p1 ?g1 ?o2 ?p2)
  )
  (:stream test-cfree-traj-pose
    :inputs (?at ?o2 ?p2)
    :domain (and (ATraj ?at) (WorldPose ?o2 ?p2))
    :certified (CFreeTrajPose ?at ?o2 ?p2)
  )

  (:stream test-gripper
    :inputs (?gq)
    :domain (GConf ?gq)
    :certified (OpenGConf ?gq)
  )
  (:stream test-door
    :inputs (?j ?a ?s)
    :domain (and (Angle ?j ?a) (Status ?s))
    :certified (AngleWithin ?j ?a ?s)
  )

  ;(:function (Distance ?bq1 ?bq2)
  ;  (and (BConf ?bq1) (BConf ?bq2))
  ;)
  ;(:function (MoveCost ?t)
  ;  (and (BTraj ?t))
  ;)
)