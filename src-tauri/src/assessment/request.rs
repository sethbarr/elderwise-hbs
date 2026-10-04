use serde_json::{json, Map, Value};

use super::EngineError;

pub const MODEL: &str = "ggml-org/Clef-Flash-GGUF:Q4_K_M";
pub const ROUTES: [&str; 4] = [
    "routine",
    "caregiver_review",
    "clinical_review",
    "urgent_review",
];
pub const URGENCY: [&str; 4] = ["Routine", "Soon", "Same day", "Immediate"];
const POLICY: &str = "Experimental wellness human-review routing only, not diagnosis or emergency triage. Scores are uncalibrated model preferences, not medical risk. No approved clinical thresholds exist. Missing/null modules are unknown, not normal. Poor/unavailable quality is unreliable: do not infer disease from it; consider human review of measurement quality. Eye metrics use normalized screen units, not degrees; no population norms. Bands are app wellness signals, not clinical findings. No automated actions.";

fn enum_value(value: &Value, allowed: &[&str]) -> Result<Value, EngineError> {
    if value.as_str().is_some_and(|s| allowed.contains(&s)) {
        Ok(value.clone())
    } else {
        Err(EngineError::new(
            "invalid_input",
            "Invalid structured assessment category",
        ))
    }
}
fn metrics(source: &Value, keys: &[&str]) -> Result<Map<String, Value>, EngineError> {
    keys.iter()
        .map(|key| {
            let v = &source[*key];
            if v.is_null() || v.as_f64().is_some_and(|n| n.is_finite() && n.abs() < 1e9) {
                Ok(((*key).to_string(), v.clone()))
            } else {
                Err(EngineError::new(
                    "invalid_input",
                    "Invalid assessment metric",
                ))
            }
        })
        .collect()
}
fn module(source: &Value, keys: &[&str]) -> Result<Value, EngineError> {
    if source.is_null() {
        return Ok(Value::Null);
    }
    if !source.is_object() {
        return Err(EngineError::new(
            "invalid_input",
            "Invalid assessment module",
        ));
    }
    let mut result = metrics(source, keys)?;
    result.insert(
        "quality".into(),
        enum_value(&source["quality"], &["good", "fair", "poor", "unavailable"])?,
    );
    result.insert(
        "band".into(),
        enum_value(&source["band"], &["good", "moderate", "limited"])?,
    );
    Ok(Value::Object(result))
}

/// Projection of the canonical Zod-validated session, not a second session contract.
/// Whitelisting also prevents narrative/transcript prompt injection and bounds context.
pub fn build_request(session: &Value) -> Result<Value, EngineError> {
    let age = &session["participant"]["age"];
    if !age.is_null() && !age.as_u64().is_some_and(|a| (18..=120).contains(&a)) {
        return Err(EngineError::new("invalid_input", "Invalid participant age"));
    }
    let mut voice = module(&session["voice"], &[])?;
    if !voice.is_null() {
        voice["markers"] = Value::Object(metrics(
            &session["voice"]["markers"],
            &[
                "f0MeanHz",
                "f0SdHz",
                "jitterPct",
                "shimmerPct",
                "hnrDb",
                "speechRateSylPerS",
                "articulationRateSylPerS",
                "pauseRatio",
                "pauseCount",
                "voicedRatio",
            ],
        )?);
    }
    let vitals = module(
        &session["vitals"],
        &[
            "heartRateBpm",
            "hrvRmssdMs",
            "hrvSdnnMs",
            "respiratoryRateBpm",
            "signalToNoise",
        ],
    )?;
    let mut eye = module(&session["eye"], &[])?;
    if !eye.is_null() {
        let tasks = session["eye"]["tasks"]
            .as_array()
            .filter(|t| t.len() <= 3)
            .ok_or_else(|| EngineError::new("invalid_input", "Invalid eye tasks"))?;
        let mut projected = Vec::new();
        for task in tasks {
            let mut m = metrics(
                task,
                &[
                    "trackingCoverage",
                    "fixationStability",
                    "saccadeCount",
                    "meanSaccadeLatencyMs",
                    "meanSaccadePeakVelocity",
                    "saccadeAccuracy",
                    "pursuitGain",
                    "blinkRatePerMin",
                ],
            )?;
            m.insert(
                "task".into(),
                enum_value(&task["task"], &["fixation", "prosaccade", "smooth-pursuit"])?,
            );
            projected.push(Value::Object(m));
        }
        eye["tasks"] = json!(projected);
    }
    Ok(json!({
        "state": {
            "participant": {"age":age,"sex":enum_value(&session["participant"]["sex"], &["female","male","unspecified"])?},
            "voice":voice,"vitals":vitals,"eye":eye,
            "overallBand":enum_value(&session["overallBand"], &["good","moderate","limited"])?
        },
        "questions": {
            "route": {"type":"choice","instructions":POLICY,"criteria":{
                "routine":"Routine wellness follow-up with available reliable signals.",
                "caregiver_review":"A trusted person reviews incomplete measurements or helps repeat the check-in.",
                "clinical_review":"A qualified professional reviews the structured findings in context.",
                "urgent_review":"Prioritize human review of concerns; this label does not establish an emergency."
            }},
            "safety_concern":{"type":"noul","instructions":format!("{POLICY} Assess whether the available reliable findings merit safety-focused human review.")},
            "urgency":{"type":"score","instructions":format!("{POLICY} Rank priority of human review, not a medical recommendation or required timeframe."),"criteria":URGENCY}
        }
    }))
}
