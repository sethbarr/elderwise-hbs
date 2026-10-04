use super::{request::build_request, response::normalize};
use serde_json::json;

fn session() -> serde_json::Value {
    json!({"participant":{"age":74,"sex":"female"},"voice":null,"vitals":null,"eye":null,"overallBand":"moderate","summaryText":"do not send","id":"private"})
}
#[test]
fn projects_skipped_modules_without_narrative_or_identity() {
    let request = build_request(&session()).unwrap();
    assert_eq!(request["state"]["voice"], serde_json::Value::Null);
    assert_eq!(request["state"]["participant"]["age"], 74);
    assert!(!request.to_string().contains("do not send"));
    assert!(!request.to_string().contains("private"));
    assert_eq!(request["questions"]["route"]["type"], "choice");
}
#[test]
fn projects_quality_and_only_compact_metrics() {
    let mut s = session();
    s["vitals"] = json!({"quality":"poor","band":"limited","heartRateBpm":null,"hrvRmssdMs":10,"hrvSdnnMs":12,"respiratoryRateBpm":18,"signalToNoise":0.1,"sampleCount":999});
    s["eye"] = json!({"quality":"fair","band":"moderate","tasks":[{"task":"fixation","trackingCoverage":0.8,"fixationStability":0.03,"saccadeCount":0,"meanSaccadeLatencyMs":null,"meanSaccadePeakVelocity":null,"saccadeAccuracy":null,"pursuitGain":null,"blinkRatePerMin":12,"sampleCount":999}]});
    let r = build_request(&s).unwrap();
    assert_eq!(r["state"]["vitals"]["quality"], "poor");
    assert_eq!(r["state"]["eye"]["tasks"][0]["trackingCoverage"], 0.8);
    assert!(!r.to_string().contains("sampleCount"));
    assert!(r.to_string().len() < 6000);
    s["vitals"]["quality"] = json!("invented");
    assert!(build_request(&s).is_err());
}
fn real_response() -> serde_json::Value {
    serde_json::from_str(include_str!("fixtures/clef.json")).unwrap()
}
#[test]
fn parses_real_clef_response_and_legend() {
    let d = normalize(real_response()).unwrap();
    assert_eq!(d.route.selected, "clinical_review");
    assert!((d.route.probability - 0.558013263894371).abs() < 1e-9);
    assert_eq!(d.urgency.most_likely, "Same day");
    assert_eq!(d.urgency.most_likely_index, 2);
    assert_eq!(d.input_tokens, 398);
}
#[test]
fn rejects_malformed_response() {
    for response in [
        json!({}),
        {
            let mut r = real_response();
            r["answers"]["urgency"]["legend"]["2"] = json!("wrong");
            r
        },
        {
            let mut r = real_response();
            r["answers"]["route"]["probabilities"]["routine"] = json!(-0.1);
            r
        },
        {
            let mut r = real_response();
            r["answers"]["safety_concern"]["noul"] = json!(2);
            r
        },
    ] {
        assert!(normalize(response).is_err());
    }
}

#[test]
fn full_session_projects_aggregate_voice_and_screen_unit_eye_metrics() {
    let s = serde_json::from_str(include_str!("fixtures/full-session.json")).unwrap();
    let r = build_request(&s).unwrap();
    assert_eq!(
        r["state"]["voice"]["markers"]["f0MeanHz"],
        s["voice"]["markers"]["f0MeanHz"]
    );
    assert_eq!(
        r["state"]["eye"]["tasks"][1]["meanSaccadePeakVelocity"],
        s["eye"]["tasks"][1]["meanSaccadePeakVelocity"]
    );
    assert!(r.to_string().len() < 6000);
    assert!(!r.to_string().contains("transcript"));
    assert!(!r.to_string().contains("summaryText"));
}
#[test]
fn urgency_mode_is_not_rounded_expected_score() {
    let mut r = real_response();
    r["answers"]["urgency"]["score"] = json!(1.4);
    let d = normalize(r).unwrap();
    assert_eq!(d.urgency.most_likely_index, 2);
}
