use super::{
    request::{ROUTES, URGENCY},
    EngineError,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AssessmentDecision {
    pub model: String,
    pub route: RouteDecision,
    pub safety: SafetyDecision,
    pub urgency: UrgencyDecision,
    pub input_tokens: u64,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RouteDecision {
    pub selected: String,
    pub probability: f64,
    pub confidence: f64,
    pub probabilities: BTreeMap<String, f64>,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SafetyDecision {
    pub concern_probability: f64,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UrgencyDecision {
    pub score: f64,
    pub most_likely: String,
    pub most_likely_index: usize,
    pub confidence: f64,
    pub probabilities: BTreeMap<String, f64>,
}
#[derive(Deserialize)]
struct Response {
    model: String,
    answers: Answers,
    usage: Usage,
}
#[derive(Deserialize)]
struct Usage {
    input_tokens: u64,
}
#[derive(Deserialize)]
struct Answers {
    route: Choice,
    safety_concern: Noul,
    urgency: Score,
}
#[derive(Deserialize)]
#[serde(tag = "type", rename = "choice")]
struct Choice {
    choice: String,
    probabilities: BTreeMap<String, f64>,
    confidence: f64,
}
#[derive(Deserialize)]
#[serde(tag = "type", rename = "noul")]
struct Noul {
    noul: f64,
}
#[derive(Deserialize)]
#[serde(tag = "type", rename = "score")]
struct Score {
    score: f64,
    legend: BTreeMap<String, String>,
    probabilities: BTreeMap<String, f64>,
    confidence: f64,
}
fn malformed() -> EngineError {
    EngineError::new(
        "invalid_response",
        "Local model returned an invalid decision",
    )
}
fn probability(n: f64) -> Result<f64, EngineError> {
    if n.is_finite() && (0.0..=1.0).contains(&n) {
        Ok(n)
    } else {
        Err(malformed())
    }
}
fn distribution(
    mut p: BTreeMap<String, f64>,
    keys: &[&str],
) -> Result<BTreeMap<String, f64>, EngineError> {
    if p.len() != keys.len() || keys.iter().any(|k| !p.contains_key(*k)) {
        return Err(malformed());
    }
    for n in p.values() {
        probability(*n)?;
    }
    let sum: f64 = p.values().sum();
    if (sum - 1.0).abs() > 0.01 {
        return Err(malformed());
    }
    for n in p.values_mut() {
        *n /= sum;
    }
    Ok(p)
}
pub fn normalize(raw: Value) -> Result<AssessmentDecision, EngineError> {
    let r: Response = serde_json::from_value(raw).map_err(|_| malformed())?;
    if r.model.is_empty() || r.model.len() > 256 {
        return Err(malformed());
    }
    let route = r.answers.route;
    let probabilities = distribution(route.probabilities, &ROUTES)?;
    let selected_probability = *probabilities.get(&route.choice).ok_or_else(malformed)?;
    if probabilities
        .values()
        .any(|p| *p > selected_probability + 1e-9)
    {
        return Err(malformed());
    }
    let urgency = r.answers.urgency;
    let urgency_p = distribution(urgency.probabilities, &["0", "1", "2", "3"])?;
    if urgency.legend.len() != 4
        || URGENCY
            .iter()
            .enumerate()
            .any(|(i, l)| urgency.legend.get(&i.to_string()).map(String::as_str) != Some(*l))
        || !urgency.score.is_finite()
        || !(0.0..=3.0).contains(&urgency.score)
    {
        return Err(malformed());
    }
    // Select the mode, not rounded expected score. Ties resolve to the first index.
    let most_likely_index = (0..4).fold(0, |best, i| {
        if urgency_p[&i.to_string()] > urgency_p[&best.to_string()] {
            i
        } else {
            best
        }
    });
    Ok(AssessmentDecision {
        model: r.model,
        route: RouteDecision {
            selected: route.choice,
            probability: selected_probability,
            confidence: probability(route.confidence)?,
            probabilities,
        },
        safety: SafetyDecision {
            concern_probability: probability(r.answers.safety_concern.noul)?,
        },
        urgency: UrgencyDecision {
            score: urgency.score,
            most_likely: URGENCY[most_likely_index].into(),
            most_likely_index,
            confidence: probability(urgency.confidence)?,
            probabilities: urgency_p,
        },
        input_tokens: r.usage.input_tokens,
    })
}
