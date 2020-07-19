from django.shortcuts import render
import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions, CategoriesOptions,SentimentOptions
import Algorithmia

from ibm_watson import ToneAnalyzerV3
# Create your views here.




client = Algorithmia.client("simpaO6Ffc92MjFpenGPg9DnlE61")

def get_content(link):
    algo = client.algo("util/Url2Text/0.1.4")
    # Limit content extracted to only blog articles
    content = algo.pipe(link).result
    return content

def main(request):
    query = request.GET.get("q")
    args = {}
    if query:
        content = get_content(query)
        input = f'{content}'
        algo = client.algo('SummarAI/Summarizer/0.1.3')
        algo.set_options(timeout=300) # optional
        summary = algo.pipe(input).result
        args['summary']=summary['summarized_data']
        authenticator = IAMAuthenticator('o48P8tGlhPPecmxPmu_autXoYp4U13mnb7dggkkiyk22')
        natural_language_understanding = NaturalLanguageUnderstandingV1(
            version='2019-07-12',
            authenticator=authenticator)
        natural_language_understanding.set_service_url("https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/972bec20-f75a-46fd-bdbc-9840fb7f7b16")

        response = natural_language_understanding.analyze(url=f'{query}' , features=Features(
                entities=EntitiesOptions(emotion=True, sentiment=True, limit=5),
                categories=CategoriesOptions(limit=1), sentiment=SentimentOptions(targets=summary['auto_gen_ranked_keywords']))).get_result()
        args['category'] = response['categories'][0]['label']
        args['category'] = args['category'].replace("/", ", ")
        args['category_score'] = response['categories'][0]['score']
        args['category_score'] = f"{(args['category_score']*100)}%"
        args['targets'] = response['sentiment']['targets']
        args['content_sentiment'] = response['sentiment']['document']['label']
        print(json.dumps(response, indent=2))

    return render(request, 'index.html', args)








