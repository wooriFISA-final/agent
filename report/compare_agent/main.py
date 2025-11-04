from change_agent_executer import build_graph

if __name__ == "__main__":
    app = build_graph()

    # test용 initial state
    test_initial_state = {
        "member_id": 1,
        "is_test": True,  # ✅ 여기를 True로 두면 테스트 모드 실행
        "report_data": None,
        "house_info": None,
        "policy_info": None,
        "credit_info": None,
        "comparison_result": None,
    }

    result = app.invoke(test_initial_state)
    print("\n🧾 최종 비교 결과:\n", result["comparison_result"])
