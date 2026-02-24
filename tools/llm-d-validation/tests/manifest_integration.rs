use llm_d_validation::{
    validate_all, validate_epp_deployment, validate_httproute, validate_inference_model,
    validate_inference_pool, validate_manifest_file,
};

#[test]
fn test_actual_inferencepool_manifest_valid() {
    let doc = validate_manifest_file("base/inferencepool.yaml").unwrap();
    validate_inference_pool(&doc).unwrap();
}

#[test]
fn test_actual_inferencemodel_manifest_valid() {
    let pool = validate_manifest_file("base/inferencepool.yaml").unwrap();
    let model = validate_manifest_file("base/inferencemodel.yaml").unwrap();
    validate_inference_model(&model, Some(&pool.metadata.name)).unwrap();
}

#[test]
fn test_actual_epp_deployment_manifest_valid() {
    let doc = validate_manifest_file("base/epp-deployment.yaml").unwrap();
    validate_epp_deployment(&doc).unwrap();
}

#[test]
fn test_actual_httproute_manifest_valid() {
    let doc = validate_manifest_file("base/httproute.yaml").unwrap();
    validate_httproute(&doc).unwrap();
}

#[test]
fn test_actual_full_stack_validates() {
    let pool = validate_manifest_file("base/inferencepool.yaml").unwrap();
    let model = validate_manifest_file("base/inferencemodel.yaml").unwrap();
    let epp = validate_manifest_file("base/epp-deployment.yaml").unwrap();
    let route = validate_manifest_file("base/httproute.yaml").unwrap();

    let errors = validate_all(&pool, &model, &epp, &route);
    assert!(errors.is_empty(), "validation errors: {errors:?}");
}

#[test]
fn test_actual_manifests_cross_reference_consistency() {
    let pool = validate_manifest_file("base/inferencepool.yaml").unwrap();
    let model = validate_manifest_file("base/inferencemodel.yaml").unwrap();
    let epp = validate_manifest_file("base/epp-deployment.yaml").unwrap();
    let route = validate_manifest_file("base/httproute.yaml").unwrap();

    let pool_name = &pool.metadata.name;
    assert_eq!(pool_name, "llm-pool");

    // Model's poolRef.name must match pool name
    let model_pool_ref = model
        .spec
        .get("poolRef")
        .and_then(|p| p.get("name"))
        .and_then(|n| n.as_str())
        .expect("model should have poolRef.name");
    assert_eq!(model_pool_ref, pool_name);

    // EPP's POOL_NAME env var must match pool name
    let containers = epp
        .spec
        .get("template")
        .and_then(|t| t.get("spec"))
        .and_then(|s| s.get("containers"))
        .and_then(|c| c.as_sequence())
        .expect("epp should have containers");
    let env_vars = containers[0]
        .get("env")
        .and_then(|e| e.as_sequence())
        .expect("epp container should have env");
    let pool_name_env = env_vars
        .iter()
        .find(|e| e.get("name").and_then(|n| n.as_str()) == Some("POOL_NAME"))
        .and_then(|e| e.get("value"))
        .and_then(|v| v.as_str())
        .expect("POOL_NAME env var should exist");
    assert_eq!(pool_name_env, pool_name);

    // HTTPRoute's backendRef name must match pool name
    let rules = route
        .spec
        .get("rules")
        .and_then(|r| r.as_sequence())
        .expect("route should have rules");
    let backend_ref_name = rules[0]
        .get("backendRefs")
        .and_then(|b| b.as_sequence())
        .and_then(|b| b.first())
        .and_then(|r| r.get("name"))
        .and_then(|n| n.as_str())
        .expect("route should have backendRef name");
    assert_eq!(backend_ref_name, pool_name);
}
