from SPARQLWrapper import SPARQLWrapper, JSON
from dataclasses import dataclass

class SPARQLManager():
    def __init__(self):
        self.ehri_sparql_endpoint = SPARQLWrapper("https://lod.ehri-project-test.eu/sparql")
        self.ehri_sparql_endpoint.setReturnFormat(JSON)

    def generate_chunk_institution_from_template(self, row):
        return f"""Institution Name: {row["name"]["value"]}
{"General Description: "+ row["description"]["value"] if "description" in row else ""}
{"History: " + row["history"]["value"] if "history" in row else ""}
{"Finding Aids: " + row["findingAids"]["value"] if "findingAids" in row else ""}
{"General Context: " + row["generalContext"]["value"] if "generalContext" in row else ""}
{"Website: " + row["website"]["value"] if "website" in row else ""}
{"Address: " + row["address"]["value"] if "address" in row else ""}
{"Opening Hours: " + row["openingHours"]["value"] if "openingHours" in row else ""}
{"City: " + row["city"]["value"] if "city" in row else ""}
{"Country: " + row["country"]["value"] if "country" in row else ""}"""
    
    def generate_chunk_archival_description_from_template(self, row):
        return f"""Archival Description Title: {row["title"]["value"]}
{"Scope and Content: "+ row["scopeAndContent"]["value"] if "scopeAndContent" in row else ""}
{"Extent: " + row["recordResourceExtent"]["value"] if "recordResourceExtent" in row else ""}
{"Structure: " + row["recordResourceStructure"]["value"] if "recordResourceStructure" in row else ""}
{"Dates: " + row["date"]["value"] if "date" in row else ""}
{"Beginning Date: " + row["beginningDate"]["value"] if "beginningDate" in row else ""}
{"End Date: " + row["endDate"]["value"] if "endDate" in row else ""}
{"General Context: " + row["generalContext"]["value"] if "generalContext" in row else ""}
{"Website: " + row["website"]["value"] if "website" in row else ""}
{"Conditions of Access: " + row["conditionsOfAccess"]["value"] if "conditionsOfAccess" in row else ""}
{"Conditions of Use: " + row["conditionsOfUse"]["value"] if "conditionsOfUse" in row else ""}
{"History: " + row["history"]["value"] if "history" in row else ""}
{"Holding Archive: " + row["archiveName"]["value"] if "archiveName" in row else ""}"""

class RagSPARQLManager(SPARQLManager):
    def __init__(self):
        super().__init__()

    def load_countries_chunks(self):
        print("Retrieving countries from SPARQL endpoint...")
        with open("conf/sparql/countries.rq", "r", encoding="utf-8") as f:
            self.ehri_sparql_endpoint.setQuery(f.read())
        result = self.ehri_sparql_endpoint.queryAndConvert()
        chunks = [chunk for row in result["results"]["bindings"] for chunk in row["o"]["value"].split(". ")]
        return list(map(lambda x: self.overlap_chunks(chunks, x), chunks))

    def load_institutions_chunks(self):
        print("Retrieving institutions from SPARQL endpoint...")
        with open("conf/sparql/institutions.rq", "r", encoding="utf-8") as f:
            self.ehri_sparql_endpoint.setQuery(f.read())
        result = self.ehri_sparql_endpoint.queryAndConvert()
        for row in result["results"]["bindings"]:
            yield self.generate_chunk_institution_from_template(row)
            
    def load_archival_descriptions_chunks(self, step, step_size):
        print("Retrieving archival descriptions from SPARQL endpoint...")
        with open("conf/sparql/archival_descriptions.rq", "r", encoding="utf-8") as f:
            self.ehri_sparql_endpoint.setQuery(f.read() + " LIMIT " + str(step_size) + " OFFSET " + str(step * step_size))
        result = self.ehri_sparql_endpoint.queryAndConvert()
        for row in result["results"]["bindings"]:
            yield self.generate_chunk_archival_description_from_template(row)
            
    def overlap_chunks(self, chunks, element):
        index = chunks.index(element)
        if(index == 0):
            return f"{element}. {chunks[index + 1]}"
        elif(len(chunks) - 1 == index):
            return f"{chunks[index - 1]}. {element}."
        else:
            return f"{chunks[index - 1]}. {element}. {chunks[index + 1]}"
        
class GraphRagSPARQLManager(SPARQLManager):
    def __init__(self):
        super().__init__()

    def load_entities_chunks(self, type):
        print(f"Retrieving entities for type {type} from SPARQL endpoint...")
        kg_types = {
            "countries": "ehri:Country",
            "institutions": "ehri:Institution",
            "archival_descriptions": "ehri:RecordSet",
        }
        with open("conf/sparql/entities_for_embedding.rq", "r", encoding="utf-8") as f:
            self.ehri_sparql_endpoint.setQuery(f.read().replace("<$type>", kg_types.get(type, "")))
        result = self.ehri_sparql_endpoint.queryAndConvert()
        return [Entity(row["sub"]["value"], 
                row["type"]["value"], 
                row["name"]["value"], 
                row["description"]["value"]) 
                for row in result["results"]["bindings"]]
        
    def load_relevant_data_country(self, country_uri):
        with open("conf/sparql/countries_additional_context.rq", "r", encoding="utf-8") as f:
            self.ehri_sparql_endpoint.setQuery(f.read().replace("$country_id", country_uri))
        result = self.ehri_sparql_endpoint.queryAndConvert()["results"]["bindings"][0]
        country_context = f"""Name: {result["name"]["value"]}
{"Archival History:" + result["archivalHistory"]["value"] if "archivalHistory" in result else ""}
{"Archival Situation:" + result["archivalSituation"]["value"] if "archivalSituation" in result else ""}
{"EHRI Research Summary:" + result["researchSummary"]["value"] if "researchSummary" in result else ""}
{"EHRI Research Extended:" + result["researchExtensive"]["value"] if "researchExtensive" in result else ""}
{"More information on the EHRI Portal: " + country_uri.replace("http://lod.ehri-project-test.eu/countries/", "https://portal.ehri-project.eu/countries/")}"""

        with open("conf/sparql/countries_linked_institutions.rq", "r", encoding="utf-8") as f:
            self.ehri_sparql_endpoint.setQuery(f.read().replace("$country_id", country_uri))
        results = self.ehri_sparql_endpoint.queryAndConvert()["results"]["bindings"]
        country_linked_institutions = [f"""Name: {row["institutionName"]["value"]}
More information on the EHRI Portal: {row["institution"]["value"].replace("http://lod.ehri-project-test.eu/institutions/", "https://portal.ehri-project.eu/institutions/")}""" 
            for row in results]
        
        return ("Country context:" + country_context
                + "\nInstitutions in this country with relevant Holocaust collection:\n" 
                + "\n".join(country_linked_institutions))
    
    def load_relevant_data_institution(self, institution_uri):
        with open("conf/sparql/institutions_additional_context.rq", "r", encoding="utf-8") as f:
            self.ehri_sparql_endpoint.setQuery(f.read().replace("$institution_id", institution_uri))
        result = self.ehri_sparql_endpoint.queryAndConvert()
        institution_context = self.generate_chunk_institution_from_template(result["results"]["bindings"][0])
        
        with open("conf/sparql/institutions_linked_archival_descriptions.rq", "r", encoding="utf-8") as f:
            self.ehri_sparql_endpoint.setQuery(f.read().replace("$institution_id", institution_uri))
        results = self.ehri_sparql_endpoint.queryAndConvert()["results"]["bindings"]
        institutions_linked_archival_descriptions = [f"""Name: {row["archivalDescriptionTitle"]["value"]}
More information on the EHRI Portal: {row["archivalDescription"]["value"].replace("http://lod.ehri-project-test.eu/units/", "https://portal.ehri-project.eu/units/")}""" 
            for row in results]
        
        return ("Institution context:" + institution_context
                + "\nArchival descriptions in this institution:\n"
                + "\n".join(institutions_linked_archival_descriptions))
    
    def load_relevant_data_archival_description(self, archival_description_uri):
        with open("conf/sparql/archival_descriptions_additional_context.rq", "r", encoding="utf-8") as f:
            self.ehri_sparql_endpoint.setQuery(f.read().replace("$archival_description_id", archival_description_uri))
        result = self.ehri_sparql_endpoint.queryAndConvert()
        archival_description_context = self.generate_chunk_archival_description_from_template(result["results"]["bindings"][0])
        
        with open("conf/sparql/archival_descriptions_linked_copies_and_originals.rq", "r", encoding="utf-8") as f:
            self.ehri_sparql_endpoint.setQuery(f.read().replace("$archival_description_id", archival_description_uri))
        results = self.ehri_sparql_endpoint.queryAndConvert()["results"]["bindings"]
        archival_description_copies_and_originals = [f"""{"Name:" + row["copyName"]["value"] if "copyName" in row else row["originalName"]["value"]}
{"Type:" + "copy" if "copyName" in row else "original"}
{"More information on the EHRI Portal:" + ("copy" if "copyName" in row else "original").replace("http://lod.ehri-project-test.eu/units/", "https://portal.ehri-project.eu/units/")}""" 
            for row in results]
        
        return ("Archival description context:" + archival_description_context
                + "\nCopies and originals linked to this description:\n"
                + "\n".join(archival_description_copies_and_originals))

@dataclass
class Entity:
    id: str
    type: str
    name: str
    description: str