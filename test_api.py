import requests
import json

# Replace with your deployed Vercel URL
BASE_URL = "https://insurance-4w9ujg7fo-arjuniscools-projects.vercel.app/"

def test_health_check():
    """Test if the API is running"""
    try:
        response = requests.get(f"{BASE_URL}/")
        print("✅ Health Check:", response.json())
        return True
    except Exception as e:
        print("❌ Health Check Failed:", str(e))
        return False

def test_insurance_claim():
    """Test insurance claim processing"""
    payload = {
        "query": "32-year-old female, appendectomy in Mumbai, 18-month policy, no pre-existing conditions"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/insurance-claim",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Insurance Claim Test:")
            print(f"   Decision: {result.get('decision')}")
            print(f"   Amount: {result.get('amount')}")
            print(f"   Criteria Count: {len(result.get('justification', []))}")
            return True
        else:
            print("❌ Insurance Claim Test Failed:", response.status_code, response.text)
            return False
    except Exception as e:
        print("❌ Insurance Claim Test Error:", str(e))
        return False

def test_document_qa():
    """Test document Q&A processing"""
    payload = {
        "documents": "https://hackrx.in/policies/BAJHLIP23020V012223.pdf",
        "questions": [
            "What is the waiting period for pre-existing diseases?",
            "Does the policy cover day-care procedures?"
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/document-qa",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Document Q&A Test:")
            for i, answer in enumerate(result.get('answers', [])):
                print(f"   Q{i+1}: {answer['question'][:50]}...")
                print(f"   A{i+1}: {answer['answer'][:100]}...")
            return True
        else:
            print("❌ Document Q&A Test Failed:", response.status_code, response.text)
            return False
    except Exception as e:
        print("❌ Document Q&A Test Error:", str(e))
        return False

def main():
    print("🧪 Testing Insurance Claims Processing MVP")
    print("=" * 50)
    
    # Update BASE_URL before running
    if "your-app-name" in BASE_URL:
        print("⚠️  Please update BASE_URL with your actual Vercel deployment URL")
        return
    
    tests = [
        ("Health Check", test_health_check),
        ("Insurance Claim Processing", test_insurance_claim),
        ("Document Q&A", test_document_qa)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        if test_func():
            passed += 1
        print("-" * 30)
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your MVP is working correctly.")
    else:
        print("⚠️  Some tests failed. Check your deployment and environment variables.")

if __name__ == "__main__":
    main()