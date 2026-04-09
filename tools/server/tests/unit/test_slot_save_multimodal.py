"""
Test slot save/restore with multimodal (image+text) input.

This test verifies that slot save/restore works correctly for multimodal models.
Key points:
- The KV cache (including image embeddings) is fully restored
- The server_tokens sidecar file (.stokens) is saved/loaded
- Image chunks cannot be fully reconstructed (requires original image data)
- NULL token placeholders for image positions are filtered out during restore
- New multimodal prompts after restore work without crashing
"""
import pytest
import os
import sys

# Add parent directory to path
sys.path.insert(0, '../')

# Only import if wget is available (optional dependency)
try:
    from utils import ServerProcess
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False
    pytest.skip("Test utilities not available", allow_module_level=True)


@pytest.fixture(autouse=True)
def create_server():
    """Set up fresh server for each test"""
    if not HAS_UTILS:
        pytest.skip("Test utilities not available")
    
    server = ServerProcess()
    server.model_file = "/home/cdanis/llama.cpp/tmp/tinygemma3-Q8_0.gguf"
    server.mmproj_file = "/home/cdanis/llama.cpp/tmp/mmproj-tinygemma3.gguf"
    server.model_hf_repo = None
    server.model_hf_file = None
    server.slot_save_path = "./tmp"
    server.temperature = 0.0
    server.server_slots = True
    server.server_port = 18080
    server.jinja = True
    yield server
    server.stop()


def test_slot_save_restore_multimodal(create_server):
    """Test slot save/restore with multimodal (image+text) input"""
    server = create_server
    server.start()

    # Use local HTTP server for images (HTTPS not available in test build)
    IMG_URL = "http://localhost:19000/truck.png"
    
    # Test 1: Original prompt with image
    res = server.make_request("POST", "/chat/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this:\n"},
                {"type": "image_url", "image_url": {"url": IMG_URL}}
            ]
        }],
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    content = res.body["choices"][0]["message"]["content"]
    print(f"Original response: {content[:100]}")
    assert len(content) > 0, "Model should have generated a response"
    assert res.body["usage"]["prompt_tokens"] > 10, "Should have processed image tokens"
    
    # Test 2: Save the slot
    res_save = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "test_slot_multimodal.bin"
    })
    assert res_save.status_code == 200
    assert res_save.body.get("n_saved", 0) > 0, "Should have saved tokens"
    
    # Verify files exist
    slot_dir = os.path.dirname(server.slot_save_path) or "."
    assert os.path.exists(os.path.join(slot_dir, "test_slot_multimodal.bin")), "KV cache file should exist"
    assert os.path.exists(os.path.join(slot_dir, "test_slot_multimodal.bin.stokens")), "Sidecar file should exist"
    
    # Test 3: Clear slot
    res_clear = server.make_request("POST", "/chat/completions", data={
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "Hello"}],
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res_clear.status_code == 200
    
    # Test 4: Restore the slot
    res_restore = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "test_slot_multimodal.bin"
    })
    assert res_restore.status_code == 200
    assert res_restore.body.get("n_restored", 0) > 0, "Should have restored tokens"
    print(f"Restore result: {res_restore.body}")
    
    # Test 5: Same prompt after restore (this is where the original crash occurred)
    res2 = server.make_request("POST", "/chat/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "max_tokens": 50,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this:\n"},
                {"type": "image_url", "image_url": {"url": IMG_URL}}
            ]
        }],
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res2.status_code == 200, "Second request should succeed (no crash)"
    content2 = res2.body["choices"][0]["message"]["content"]
    print(f"After restore response: {content2[:100]}")
    assert len(content2) > 0, "Model should have generated a response after restore"
    assert res2.body["usage"]["prompt_tokens"] > 10, "Should have processed image tokens after restore"
    
    # Test 6: Verify inference correctness - output should be identical
    # (deterministic with temperature=0.0, so same prompt = same output)
    assert content == content2, f"Inference should be identical after restore\nBefore: {content[:100]}\nAfter:  {content2[:100]}"
    print("✓ Inference correctness verified: identical output after restore")
    
    print("Test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
