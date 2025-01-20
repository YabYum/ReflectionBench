from pipeline import REFLECTIONBENCH


if __name__ == "__main__":
    
    # Set the model_name and Output strategy (True for CoT/ None for free output/ 'Direct' for direct generation)
    evaluator_true = REFLECTIONBENCH(model='gpt-4o-mini', COT=None)
    evaluator_true.evaluate()
