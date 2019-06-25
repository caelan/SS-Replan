(define (stream nvidia-tamp)
  (:stream sample-pose
    :inputs (?o ?r)
    :domain (Stackable ?o ?r)
    :outputs (?p)
    :certified (and (Pose ?o ?p) (Supported ?o ?p ?r))
  )
  (:stream sample-grasp
    :inputs (?o)
    :domain (Graspable ?o)
    :outputs (?g)
    :certified (Grasp ?o ?g)
  )

  ; Movable base
  (:stream inverse-kinematics
    :inputs (?o ?p ?g)
    :domain (and (Pose ?o ?p) (Grasp ?o ?g) (MovableBase))
    :outputs (?bq ?aq ?at)
    :certified (and (BConf ?bq) (AConf ?aq) (ATraj ?at)
                    (Kin ?o ?p ?g ?bq ?aq ?at))
  )
  (:stream plan-pull
    :inputs (?j ?a1 ?a2)
    :domain (and (Angle ?j ?a1) (Angle ?j ?a2) (MovableBase))
    :outputs (?bq ?aq ?at)
    :certified (and (BConf ?bq) (AConf ?aq) (ATraj ?at)
                    (Pull ?j ?a1 ?a2 ?bq ?aq ?at))
  )

  ; Fixed base
  (:stream fixed-inverse-kinematics ; TODO: check if ?p ?g in convex hull
    :inputs (?o ?p ?g ?bq)
    :domain (and (Pose ?o ?p) (Grasp ?o ?g) (InitBConf ?bq))
    :outputs (?aq ?at)
    :certified (and (AConf ?aq) (ATraj ?at)
                    (Kin ?o ?p ?g ?bq ?aq ?at))
  )
  ;(:stream fixed-plan-pull ; TODO: check if ?j within range
  ;  :inputs (?j ?a1 ?a2 ?bq)
  ;  :domain (and (Angle ?j ?a1) (Angle ?j ?a2) (InitBConf ?bq))
  ;  :outputs (?bq ?aq ?at)
  ;  :certified (and (BConf ?bq) (AConf ?aq) (ATraj ?at)
  ;                  (Pull ?j ?a1 ?a2 ?bq ?aq ?at))
  ;)

  (:stream plan-base-motion
    :fluents (AtPose AtGrasp AtAngle)
    :inputs (?bq1 ?bq2)
    :domain (and (BConf ?bq1) (BConf ?bq2) (MovableBase))
    :outputs (?bt)
    :certified (and (BTraj ?bt) (BaseMotion ?bq1 ?bq2 ?bt))
  )
  (:stream plan-calibrate-motion
    :inputs (?bq)
    :domain (BConf ?bq)
    :outputs (?aq ?at)
    :certified (and (AConf ?aq) (ATraj ?at)
                    (CalibrateMotion ?bq ?aq ?at))
  )

  (:stream test-cfree-pose-pose
    :inputs (?o1 ?p1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Pose ?o2 ?p2))
    :certified (CFreePosePose ?o1 ?p1 ?o2 ?p2)
  )
  (:stream test-cfree-approach-pose
    :inputs (?o1 ?p1 ?g1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Grasp ?o1 ?g1) (Pose ?o2 ?p2))
    :certified (CFreeApproachPose ?o1 ?p1 ?g1 ?o2 ?p2)
  )
  (:stream test-cfree-approach-angle
    :inputs (?o1 ?p1 ?g1 ?j ?a)
    :domain (and (Pose ?o1 ?p1) (Grasp ?o1 ?g1) (Angle ?j ?a))
    :certified (CFreeApproachAngle ?o1 ?p1 ?g1 ?j ?a)
  )
  (:stream test-cfree-traj-pose
    :inputs (?at ?o2 ?p2)
    :domain (and (ATraj ?at) (Pose ?o2 ?p2))
    :certified (CFreeTrajPose ?at ?o2 ?p2)
  )
  (:stream test-cfree-traj-angle
    :inputs (?at ?j ?a)
    :domain (and (ATraj ?at) (Angle ?j ?a))
    :certified (CFreeTrajAngle ?at ?j ?a)
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