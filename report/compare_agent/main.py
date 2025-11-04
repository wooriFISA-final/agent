from report.compare_agent.change_agent_executer import build_graph

if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({})
    print("\n🧾 최종 비교 결과:\n", result["comparison_result"])
