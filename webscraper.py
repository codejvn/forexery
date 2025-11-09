import asyncio
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv
from dedalus_labs.utils.streaming import stream_async
import csv, io

load_dotenv()

async def main():
    print("Starting web scraping for AI agent developments...")
    client = AsyncDedalus()
    runner = DedalusRunner(client)

    total_years = 5
    quarter_begin = 3
    quarter_end = 4
    year = 2025
    newYear = False
    country = "America"
    for _ in range(total_years*2):
        result = await runner.run(
            input=f"""

            I'm trying to create a CSV using the headlines from
            reputable news articles about finance, finance-related policies,
            and major national-level political news in {country}.
            
            I need daily headlines from Q{quarter_begin} to Q{quarter_end} in {year}. 
            I want the data to be returned to me as a text block, with each row representing 
            an article, and different columns of information for each article separated 
            by a comma. 

            Please help me:
            1. Find headlines of articles covering finance, policies, and politics
            from Q{quarter_begin} to Q{quarter_end} in {year}
            2. Record dateline of publication (YYYY-MM-DD) and the headline of the article in a row. 
            3. Parse through the headline, and determine if the headline is positive, negative, or neutral in sentiment.
            4. Depending on the sentiment, add a final column to the row indicating "Positive", "Negative", or "Neutral".
            5. In total, a given row should have 3 columns: Dateline, Headline, Sentiment, all for the same article.
            6. Repeat this so that there is one article for every day in the given time period.
            
            Return the data as a text block, with each row representing an article, and different columns of information 
            for each article separated by a comma. Limit the output to only the raw CSV data, without any additional commentary or explanation.
            If there are limitations with the search tools, make multiple queries to fulfill the requirements I specified.
            If the CSV is being received in multiple parts, ensure that each part does not include the header row.
            I prefer the everything to be in one message, even if it is a very long message, and not 
            separated in any way. 
            The sources of artciles should be reputable, but I don't have a set list of sources that you are limited to.
            Use whatever sources you think are best, do not ask me for websites to use. 
            Sentiment should be based on the headline text only to save processing time. 
            For Q4 2025, I wasnt daily rows from 2025-07-01 to now. 
            Unless you have a serious outstanding doubt, don't ask me any clarifying questions, just proceed with the task
            and use your best judgement given the context of the prompt to solve any minor questions. 
            """,
            model="openai/gpt-5",
            mcp_servers=[
                 "joerup/exa-mcp",        # Semantic search engine 
                #"windsor/brave-search-mcp"  # Privacy-focused web search
            ]
        )
        #If you, for whatever reason, cannot fulfill this request, return an empty CSV with no header row
        
        print(f"{result.output}")
        csv_input = result.output.strip()
        lines = csv_input.split('\n')
        csv_input = '\n'.join(lines[1:])  # Skip header if present
        if country == "America":
            append_csv_string_to_file(csv_input, 'news_articles_usa.csv')
            country = "Japan"
        else:
            append_csv_string_to_file(csv_input, 'news_articles_japan.csv')
            country = "America"
            if newYear == False:
                quarter_begin = 1
                quarter_end = 2
                newYear = True
            else:
                year -= 1
                quarter_begin = 3
                quarter_end = 4
                newYear = False



def append_csv_string_to_file(csv_string_data: str, filename: str):
    """
    Appends CSV data provided as a string to an existing CSV file.

    Args:
        csv_string_data: A string containing one or more rows of CSV data.
        filename: The path to the existing CSV file to append to.
    """
    try:
        # Use io.StringIO to treat the input string as a file
        string_io = io.StringIO(csv_string_data)
        
        # Read the rows from the input string
        reader = csv.reader(string_io)
        new_rows = list(reader)

        # Open the target file in append mode ('a')
        # newline='' is essential for proper CSV handling across all operating systems
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # Append the new rows
            writer.writerows(new_rows) # Use writerows for multiple rows
        
        print(f"Successfully appended {len(new_rows)} row(s) to {filename}")

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())