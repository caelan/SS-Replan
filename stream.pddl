(define (stream nvidia-tamp)
  (:stream sample-pose
    :inputs (?o ?r)
    :domain (Stackable ?o ?r)
    :outputs (?rp)
    :certified (RelPose ?o ?rp ?r)
  )
  (:stream sample-grasp
    :inputs (?o)
    :domain (Graspable ?o)
    :outputs (?g)
    :certified (Grasp ?o ?g)
  )

  ; Movable base
  (:stream plan-pick
    :inputs (?o ?p ?g)
    :domain (and (WorldPose ?o ?p) (Grasp ?o ?g) (MovableBase))
    :outputs (?bq ?at)
    :certified (and (BConf ?bq) (ATraj ?at) ; (AConf ?aq)
                    (Pick ?o ?p ?g ?bq ?at))
  )
  (:stream plan-pull
    :inputs (?j ?a1 ?a2)
    :domain (and (Angle ?j ?a1) (Angle ?j ?a2) (MovableBase))
    :outputs (?bq ?at)
    :certified (and (BConf ?bq) (ATraj ?at) ; (AConf ?aq)
                    (Pull ?j ?a1 ?a2 ?bq ?at))
  )

  ; Fixed base
  (:stream fixed-plan-pick ; TODO: check if ?p ?g in convex hull
    :inputs (?o ?p ?g ?bq)
    :domain (and (WorldPose ?o ?p) (Grasp ?o ?g) (InitBConf ?bq))
    :outputs (?at)
    :certified (and (ATraj ?at) ; (AConf ?aq)
                    (Pick ?o ?p ?g ?bq ?at))
  )
  (:stream fixed-plan-pull ; TODO: check if ?j within range
    :inputs (?j ?a1 ?a2 ?bq)
    :domain (and (Angle ?j ?a1) (Angle ?j ?a2) (InitBConf ?bq))
    :outputs (?at)
    :certified (and (ATraj ?at) ; (AConf ?aq)
                    (Pull ?j ?a1 ?a2 ?bq ?at))
  )

  (:stream plan-base-motion
    :fluents (AtWorldPose AtGrasp) ; AtAngle
    :inputs (?bq1 ?bq2)
    :domain (and (BConf ?bq1) (BConf ?bq2) (MovableBase))
    :outputs (?bt)
    :certified (and ; (BTraj ?bt)
                    (BaseMotion ?bq1 ?bq2 ?bt))
  )
  (:stream plan-arm-motion
    :fluents (AtBConf AtWorldPose AtGrasp) ; AtAngle
    :inputs (?aq1 ?aq2)
    :domain (and (AConf ?aq1) (AConf ?aq2))
    :outputs (?at)
    :certified (and ; (ATraj ?bt)
                    (ArmMotion ?aq1 ?aq2 ?at))
  )
  (:stream plan-calibrate-motion
    :inputs (?bq)
    :domain (BConf ?bq)
    :outputs (?at)
    :certified (and (ATraj ?at) ; (AConf ?aq)
                    (CalibrateMotion ?bq ?at))
  )

  (:stream compute-pose-kin
    :inputs (?o1 ?rp ?o2 ?p2)
    :domain (and (RelPose ?o1 ?rp ?o2) (WorldPose ?o2 ?p2))
    :outputs (?p1)
    :certified (and (WorldPose ?o1 ?p1) ; (PoseTriplet ?o1 ?p1 ?rp) ; For instantiation
                    (PoseKin ?o1 ?p1 ?rp ?o2 ?p2))
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

  (:stream test-door
    :inputs (?j ?a ?s)
    :domain (and (Angle ?j ?a) (Status ?s))
    :certified (AngleWithin ?j ?a ?s)
  )

  (:function (Distance ?bq1 ?bq2)
    (and (BConf ?bq1) (BConf ?bq2))
  )
  ;(:function (MoveCost ?t)
  ;  (and (BTraj ?t))
  ;)
)