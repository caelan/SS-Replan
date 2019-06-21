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
  (:stream inverse-kinematics
    :inputs (?o ?p ?g)
    :domain (and (Pose ?o ?p) (Grasp ?o ?g))
    :outputs (?bq ?aq ?at)
    :certified (and (BConf ?bq) (AConf ?aq) (ATraj ?at)
                    (Kin ?o ?p ?g ?bq ?aq ?at))
  )
  (:stream plan-pull
    :inputs (?j ?a1 ?a2)
    :domain (and (Angle ?j ?a1) (Angle ?j ?a2))
    :outputs (?bq ?aq ?at)
    :certified (and (BConf ?bq) (AConf ?aq) (ATraj ?at)
                    (Pull ?j ?a1 ?a2 ?bq ?aq ?at))
  )
  (:stream plan-base-motion
    :fluents (AtPose AtGrasp AtAngle)
    :inputs (?bq1 ?bq2)
    :domain (and (BConf ?bq1) (BConf ?bq2))
    :outputs (?bt)
    :certified (and (BTraj ?bt) (BaseMotion ?bq1 ?bq2 ?bt))
  )

;  (:stream test-cfree-pose-pose
;    :inputs (?o1 ?p1 ?o2 ?p2)
;    :domain (and (Pose ?o1 ?p1) (Pose ?o2 ?p2))
;    :certified (CFreePosePose ?o1 ?p1 ?o2 ?p2)
;  )
;  (:stream test-cfree-approach-pose
;    :inputs (?o1 ?p1 ?g1 ?o2 ?p2)
;    :domain (and (Pose ?o1 ?p1) (Grasp ?o1 ?g1) (Pose ?o2 ?p2))
;    :certified (CFreeApproachPose ?o1 ?p1 ?g1 ?o2 ?p2)
;  )
;  (:stream test-cfree-traj-pose
;    :inputs (?t ?o2 ?p2)
;    :domain (and (ATraj ?t) (Pose ?o2 ?p2))
;    :certified (CFreeTrajPose ?t ?o2 ?p2)
;  )
  ;(:stream test-cfree-traj-grasp-pose
  ;  :inputs (?t ?a ?o1 ?g1 ?o2 ?p2)
  ;  :domain (and (BTraj ?t) (Arm ?a) (Grasp ?o1 ?g1) (Pose ?o2 ?p2))
  ;  :certified (CFreeTrajGraspPose ?t ?a ?o1 ?g1 ?o2 ?p2)
  ;)


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