import boto3
import logging

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear existing handlers to avoid duplicates
if logger.handlers:
    logger.handlers.clear()

# Custom formatter with all requested fields separated by commas
formatter = logging.Formatter(
    "%(asctime)s,%(levelname)s,%(process)d,%(filename)s,%(lineno)d,%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S.%f"
)

# Add handler with the custom formatter
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)



def get_or_create_guardrail(bedrock_client):
    guardrail_name = 'dsan-program-guardrails'

    try:
        # First, check if a guardrail with this name already exists
        existing_guardrails = bedrock_client.list_guardrails()['guardrails']
        logger.info(f"Existing guardrails: {existing_guardrails}")
        for guardrail in existing_guardrails:
            if guardrail['name'] == guardrail_name:
                logger.info(f"Guardrail already exists: {guardrail}")
                return guardrail['id'], guardrail['version']

        # If not found, create it
        response = bedrock_client.create_guardrail(
            name=guardrail_name,
            description='Ensures the chatbot provides accurate information based on DSAN program materials while maintaining academic integrity and appropriate boundaries.',
            topicPolicyConfig={
                'topicsConfig': [
                    # {
                    #     'name': 'Non-Public Information',
                    #     'definition': 'Providing information that is not available on public DSAN program webpages or making claims beyond official program documentation.',
                    #     'examples': [
                    #         'What are the private discussions in faculty meetings?',
                    #         'How many students were rejected last year?',
                    #         'What is the exact acceptance rate?',
                    #         'Can you share internal program metrics?',
                    #         'Who are the students currently enrolled in DSAN?'
                    #     ],
                    #     'type': 'DENY'
                    # },
                    {
                        'name': 'Future Predictions',
                        'definition': 'Making predictions or promises about future program changes, admissions, or course offerings not stated in official DSAN documentation.',
                        'examples': [
                            'Will the program requirements change next year?',
                            'What new data science courses will be added?',
                            'Will the program tuition increase next semester?',
                            'What will be the future job placement rate?',
                            'Are there plans to change the curriculum?'
                        ],
                        'type': 'DENY'
                    },
                    {
                        'name': 'Personal Advice',
                        'definition': 'Providing personalized recommendations or decisions that should be made by DSAN program administrators or academic advisors.',
                        'examples': [
                            'Should I choose over other data science programs?',
                            'Which electives should I take given my background?',
                            'Would I be successful in the program?',
                            'Can I handle the advanced analytics coursework?',
                        ],
                        'type': 'DENY'
                    },
                    {
                        'name': 'Academic Integrity',
                        'definition': 'Maintaining academic integrity by not providing direct answers to assignments, exam questions, or project solutions.',
                        'examples': [
                            'Can you solve this homework problem?',
                            'What are the answers to the midterm exam?',
                            'Write code for my class project.',
                            'Debug my assignment solution.',
                            'How should I answer this quiz question?'
                        ],
                        'type': 'DENY'
                    },
                    {
                        'name': 'Technical Implementation',
                        'definition': 'Providing specific technical implementation details about the program systems or infrastructure.',
                        'examples': [
                            'How is the DSAN website backend implemented?',
                            'What servers does the program use?',
                            'How are student records stored?',
                            'What is the database structure?',
                            'Share the system architecture details.'
                        ],
                        'type': 'DENY'
                    }
                ]
            },
            contentPolicyConfig={
                'filtersConfig': [
                    {'type': 'SEXUAL', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                    {'type': 'VIOLENCE', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                    {'type': 'HATE', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                    {'type': 'INSULTS', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                    {'type': 'MISCONDUCT', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                    {'type': 'PROMPT_ATTACK', 'inputStrength': 'HIGH', 'outputStrength': 'NONE'}
                ]
            },
            blockedInputMessaging = (
            "It looks like your message might contain sensitive, inappropriate, or restricted content. "
            "I’m here to help within respectful and academic boundaries. For accurate information about the DSAN program, please visit https://analytics.georgetown.edu or reach out to a program advisor."
            ),
            blockedOutputsMessaging = (
            "I can’t provide that information as it may involve sensitive topics or go beyond what I'm allowed to share. "
            "For reliable details about the DSAN program, please visit https://analytics.georgetown.edu or contact a program administrator."
            ),
        )
        logger.info(f"Guardrail created successfully: {response}")
        return response['guardrailId'], response['version']

    except Exception as e:
        logger.error(f"Failed to get or create guardrail: {str(e)}")
        raise
