import streamlit as st
from crewai import Crew
from trip_agents import TripAgents
from trip_tasks import TripTasks
import warnings
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

class TripCrew:
    def __init__(self, origin, cities, date_range, interests):
        self.cities = cities
        self.origin = origin
        self.interests = interests
        self.date_range = date_range

    def run(self):
        agents = TripAgents()
        tasks = TripTasks()

        city_selector_agent = agents.city_selection_agent()
        local_expert_agent = agents.local_expert()
        travel_concierge_agent = agents.travel_concierge()

        identify_task = tasks.identify_task(
            city_selector_agent,
            self.origin,
            self.cities,
            self.interests,
            self.date_range
        )
        gather_task = tasks.gather_task(
            local_expert_agent,
            self.origin,
            self.interests,
            self.date_range
        )
        plan_task = tasks.plan_task(
            travel_concierge_agent, 
            self.origin,
            self.interests,
            self.date_range
        )

        crew = Crew(
            agents=[
                city_selector_agent, local_expert_agent, travel_concierge_agent
            ],
            tasks=[identify_task, gather_task, plan_task],
            verbose=True
        )

        result = crew.kickoff()
        return result

# Streamlit UI
st.title("🛫 AI Trip Planner")
st.write("Plan your next trip with AI-powered assistance!")

location = st.text_input("From where will you be traveling from?")
cities = st.text_area("What are the cities options you are interested in visiting?")
date_range = st.text_input("What is the date range you are interested in traveling?")
interests = st.text_area("What are some of your high-level interests and hobbies?")
if st.button("Generate Trip Plan"):
    if location and cities and date_range and interests:
        trip_crew = TripCrew(location, cities, date_range, interests)
        result = trip_crew.run()
        
        st.subheader("Here is your AI-generated Trip Plan:")
        st.markdown(result)  # Display result in markdown format
    else:
        st.error("Please fill in all fields before generating the trip plan.")
