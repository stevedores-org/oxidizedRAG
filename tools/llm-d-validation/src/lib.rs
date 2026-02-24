use serde::Deserialize;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("failed to read file: {0}")]
    FileRead(#[from] std::io::Error),

    #[error("failed to parse YAML: {0}")]
    YamlParse(#[from] serde_yaml::Error),

    #[error("invalid manifest: {0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, ValidationError>;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct K8sDocument {
    pub api_version: String,
    pub kind: String,
    pub metadata: Metadata,
    #[serde(default)]
    pub spec: serde_yaml::Value,
}

#[derive(Debug, Deserialize)]
pub struct Metadata {
    pub name: String,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub labels: Option<serde_yaml::Value>,
}

/// Load and parse a YAML manifest from an arbitrary path.
pub fn load_manifest(path: &Path) -> Result<K8sDocument> {
    let content = std::fs::read_to_string(path)?;
    let doc: K8sDocument = serde_yaml::from_str(&content)?;
    Ok(doc)
}

/// Load a manifest relative to the crate root (using `CARGO_MANIFEST_DIR`).
pub fn validate_manifest_file(relative_path: &str) -> Result<K8sDocument> {
    let base = env!("CARGO_MANIFEST_DIR");
    let path = Path::new(base).join(relative_path);
    load_manifest(&path)
}

pub fn validate_inference_pool(doc: &K8sDocument) -> Result<()> {
    if doc.kind != "InferencePool" {
        return Err(ValidationError::Invalid(format!(
            "expected kind InferencePool, got {}",
            doc.kind
        )));
    }
    if !doc
        .api_version
        .starts_with("inference.networking.x-k8s.io/")
    {
        return Err(ValidationError::Invalid(format!(
            "unexpected apiVersion: {}",
            doc.api_version
        )));
    }

    let spec = &doc.spec;
    if spec.get("targetPortNumber").is_none() {
        return Err(ValidationError::Invalid(
            "spec.targetPortNumber is required".into(),
        ));
    }
    if spec.get("selector").is_none() {
        return Err(ValidationError::Invalid("spec.selector is required".into()));
    }
    if spec.get("extensionRef").is_none() {
        return Err(ValidationError::Invalid(
            "spec.extensionRef is required".into(),
        ));
    }

    Ok(())
}

pub fn validate_inference_model(doc: &K8sDocument, expected_pool: Option<&str>) -> Result<()> {
    if doc.kind != "InferenceModel" {
        return Err(ValidationError::Invalid(format!(
            "expected kind InferenceModel, got {}",
            doc.kind
        )));
    }
    if !doc
        .api_version
        .starts_with("inference.networking.x-k8s.io/")
    {
        return Err(ValidationError::Invalid(format!(
            "unexpected apiVersion: {}",
            doc.api_version
        )));
    }

    let spec = &doc.spec;
    if spec.get("modelName").is_none() {
        return Err(ValidationError::Invalid(
            "spec.modelName is required".into(),
        ));
    }
    if spec.get("criticality").is_none() {
        return Err(ValidationError::Invalid(
            "spec.criticality is required".into(),
        ));
    }

    let pool_ref = spec
        .get("poolRef")
        .ok_or_else(|| ValidationError::Invalid("spec.poolRef is required".into()))?;
    let pool_name = pool_ref
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ValidationError::Invalid("spec.poolRef.name is required".into()))?;

    if let Some(expected) = expected_pool {
        if pool_name != expected {
            return Err(ValidationError::Invalid(format!(
                "poolRef.name mismatch: expected {expected}, got {pool_name}"
            )));
        }
    }

    Ok(())
}

pub fn validate_epp_deployment(doc: &K8sDocument) -> Result<()> {
    if doc.kind != "Deployment" {
        return Err(ValidationError::Invalid(format!(
            "expected kind Deployment, got {}",
            doc.kind
        )));
    }

    let spec = &doc.spec;
    let template = spec
        .get("template")
        .ok_or_else(|| ValidationError::Invalid("spec.template is required".into()))?;
    let containers = template
        .get("spec")
        .and_then(|s| s.get("containers"))
        .and_then(|c| c.as_sequence())
        .ok_or_else(|| {
            ValidationError::Invalid("spec.template.spec.containers is required".into())
        })?;

    if containers.is_empty() {
        return Err(ValidationError::Invalid(
            "at least one container is required".into(),
        ));
    }

    let container = &containers[0];

    // Check required env vars
    let env_vars = container
        .get("env")
        .and_then(|e| e.as_sequence())
        .ok_or_else(|| ValidationError::Invalid("container env vars are required".into()))?;

    let env_names: Vec<&str> = env_vars
        .iter()
        .filter_map(|e| e.get("name").and_then(|n| n.as_str()))
        .collect();

    for required in &["POOL_NAME", "POOL_NAMESPACE"] {
        if !env_names.contains(required) {
            return Err(ValidationError::Invalid(format!(
                "missing required env var: {required}"
            )));
        }
    }

    // Check readiness probe
    if container.get("readinessProbe").is_none() {
        return Err(ValidationError::Invalid(
            "container readinessProbe is required".into(),
        ));
    }

    // Check resources
    if container.get("resources").is_none() {
        return Err(ValidationError::Invalid(
            "container resources are required".into(),
        ));
    }

    Ok(())
}

pub fn validate_httproute(doc: &K8sDocument) -> Result<()> {
    if doc.kind != "HTTPRoute" {
        return Err(ValidationError::Invalid(format!(
            "expected kind HTTPRoute, got {}",
            doc.kind
        )));
    }
    if !doc.api_version.starts_with("gateway.networking.k8s.io/") {
        return Err(ValidationError::Invalid(format!(
            "unexpected apiVersion: {}",
            doc.api_version
        )));
    }

    let spec = &doc.spec;
    let parent_refs = spec
        .get("parentRefs")
        .and_then(|p| p.as_sequence())
        .ok_or_else(|| ValidationError::Invalid("spec.parentRefs is required".into()))?;

    if parent_refs.is_empty() {
        return Err(ValidationError::Invalid(
            "spec.parentRefs must not be empty".into(),
        ));
    }

    let rules = spec
        .get("rules")
        .and_then(|r| r.as_sequence())
        .ok_or_else(|| ValidationError::Invalid("spec.rules is required".into()))?;

    // Check that at least one rule has backendRefs
    let has_backend_refs = rules.iter().any(|rule| {
        rule.get("backendRefs")
            .and_then(|b| b.as_sequence())
            .map(|b| !b.is_empty())
            .unwrap_or(false)
    });

    if !has_backend_refs {
        return Err(ValidationError::Invalid(
            "at least one rule must have backendRefs".into(),
        ));
    }

    Ok(())
}

/// Run all four validations and collect errors.
pub fn validate_all(
    pool: &K8sDocument,
    model: &K8sDocument,
    epp: &K8sDocument,
    route: &K8sDocument,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    if let Err(e) = validate_inference_pool(pool) {
        errors.push(e);
    }
    if let Err(e) = validate_inference_model(model, Some(&pool.metadata.name)) {
        errors.push(e);
    }
    if let Err(e) = validate_epp_deployment(epp) {
        errors.push(e);
    }
    if let Err(e) = validate_httproute(route) {
        errors.push(e);
    }

    errors
}
