package model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Properties;
import org.json.JSONObject;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.CoreMap;
import me.xdrop.fuzzywuzzy.FuzzySearch;
import me.xdrop.fuzzywuzzy.model.ExtractedResult;


public class NdaParser {
	private enum ENTITY_TYPE {
		PERSON,
		ORGANIZATION,
		LOCATION,
		NUMBER,
		ORDINAL,
		DATE
	}

	private static List<String> MISLEADING = Arrays.asList(new String[] { "NDA" });


	public static JSONObject parseForOtherPartyInfo(StanfordCoreNLP stanfordNERPipeline, String text, String userName, String userCompany) throws Exception {
		text = text.substring(0, 400) + " " + text.substring(text.length() - 400, text.length());
		JSONObject result = new JSONObject();
		JSONObject receiver = new JSONObject();
		List<CoreMap> sentences = getEntities(stanfordNERPipeline, text);
		List<String> person = new ArrayList<String>();
		List<String> organization = new ArrayList<String>();
		List<String> location = new ArrayList<String>();

		firstRound(sentences, person, organization, location);
		secondRound(sentences, location);
		removeUserName(userName, person);
		removeUserCompany(userCompany, organization);

		String title = seachForTitle(text).trim();
		result.put("title", title);
		if (person.size() > 0)
			receiver.put("name", person.get(0));
		if (organization.size() > 0)
			receiver.put("company_name", organization.get(0));
		if (location.size() > 0)
			receiver.put("company_address", location.get(0));
		result.put("receiver", receiver);

		return result;
	}


	private static void removeUserName(String makerName, List<String> person) {
		List <ExtractedResult> makerNameClones = FuzzySearch.extractTop(makerName, person, 3);
		for (int i = 0; i < makerNameClones.size(); i++) {
			ExtractedResult makerClone = makerNameClones.get(i);
			if (makerClone.getScore() > 60)
				while(person.remove(makerClone.getString()));
		}
	}


	private static void removeUserCompany(String makerCompany, List<String> organization) {
		List <ExtractedResult> makerCompanyClones = new ArrayList<ExtractedResult>();
		if (!isNullOrEmpty(makerCompany)) {
			makerCompanyClones = FuzzySearch.extractTop(makerCompany, organization, 3);
			for (int i = 0; i < makerCompanyClones.size(); i++) {
				ExtractedResult makerClone = makerCompanyClones.get(i);
				if (makerClone.getScore() > 60)
					while(organization.remove(makerClone.getString()));
			}
		}
	}


	private static String seachForTitle(String text) {
		text = text.replaceAll("\t", " ");
		text = text.replaceAll("\n", " ");
		String title = "Non-Disclosure Agreement";
		List<String> textSplit = new LinkedList<String>(Arrays.asList(text.split(" ")));
		for (int i = 0; i < textSplit.size(); i++) {
			String word = textSplit.get(i);
			if (word.length() <= 1)
				textSplit.remove(i);
		}

		for (int i = 0; i < textSplit.size(); i++) {
			String resultStr = "";
			String key = "Mutual Confidentiality and Non-Disclosure Agreement";
			ExtractedResult result = FuzzySearch.extractOne(key, Arrays.asList(new String[] { textSplit.get(i) }));
			if (result.getScore() >= 60) {
				do {
					resultStr += " " + result.getString();
					i++;
					result = FuzzySearch.extractOne(key, Arrays.asList(new String[] { textSplit.get(i) }));
				} while (result.getScore() >= 50 && i< textSplit.size() - 1);
				if (resultStr.length() > 0) {
					title = resultStr;
					return title;
				}
			}
		}

		return title;
	}


	private static void secondRound(List<CoreMap> sentences, List<String> location) {
		for (CoreMap sentence : sentences) {
			List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
			for (int i = 0; i < tokens.size(); i++) {
				CoreLabel token = tokens.get(i);
				String ne = token.get(NamedEntityTagAnnotation.class);
				if (!ne.equals(ENTITY_TYPE.LOCATION.toString()))
					continue;
				String tokenText = token.originalText();
				i++;
				if (i < tokens.size()) {
					String nextne = tokens.get(i).get(NamedEntityTagAnnotation.class);
					while (i + 1 < tokens.size() && nextne.equals(ENTITY_TYPE.LOCATION.toString())) {
						tokenText += " " + tokens.get(i).originalText();
						i++;
						nextne = tokens.get(i).get(NamedEntityTagAnnotation.class);
					}
				}
				location.add(tokenText);
			}
		}
	}


	private static void firstRound(List<CoreMap> sentences, List<String> person, List<String> organization, List<String> location) {
		for (CoreMap sentence : sentences) {
			List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
			for (int i = 0; i < tokens.size(); i++) {
				CoreLabel token = tokens.get(i);
				String ne = token.get(NamedEntityTagAnnotation.class);
				if (!neededEntity(ne))
					continue;
				ENTITY_TYPE type = ENTITY_TYPE.valueOf(ne);
				int newIndex = forPerson(tokens, type, token, i, person);
				if (newIndex > i) {
					i = newIndex;
					continue;
				}
				newIndex = forOrganization(tokens, type, token, i, organization);
				if (newIndex > i) {
					i = newIndex;
					continue;
				}
				newIndex = forAddress(tokens, type, token, i, location);
				if (newIndex > i) {
					i = newIndex;
					continue;
				}
			}
		}
	}


	private static int forPerson(List<CoreLabel> tokens, ENTITY_TYPE entityType, CoreLabel token, int i,
			List<String> person) {
		if (entityType == ENTITY_TYPE.PERSON && token.originalText().length() >= 3) {
			String tokenText = token.originalText();
			i++;
			if (i < tokens.size()) {
				String nextne = tokens.get(i).get(NamedEntityTagAnnotation.class);
				while (i + 1 < tokens.size() && nextne.equals(ENTITY_TYPE.PERSON.toString())) {
					tokenText += " " + tokens.get(i).originalText();
					i++;
					nextne = tokens.get(i).get(NamedEntityTagAnnotation.class);
				}
			}
			person.add(tokenText);
		}
		return i;
	}


	private static int forOrganization(List<CoreLabel> tokens, ENTITY_TYPE type, CoreLabel token, int i,
			List<String> organization) {
		String tokenText = token.originalText();
		if (type == ENTITY_TYPE.ORGANIZATION && token.originalText().length() >= 3 && ! MISLEADING.contains(tokenText)) {
			i++;
			if (i < tokens.size()) {
				String nextne = tokens.get(i).get(NamedEntityTagAnnotation.class);
				while (i + 1 < tokens.size() && nextne.equals(ENTITY_TYPE.ORGANIZATION.toString())) {
					tokenText += " " + tokens.get(i).originalText();
					i++;
					nextne = tokens.get(i).get(NamedEntityTagAnnotation.class);
				}
			}
			organization.add(tokenText);
		}

		return i;
	}


	private static int forAddress(List<CoreLabel> tokens, ENTITY_TYPE type, CoreLabel token, int i,
			List<String> location) {
		if (type == ENTITY_TYPE.NUMBER || type == ENTITY_TYPE.ORDINAL || type == ENTITY_TYPE.DATE
				|| type == ENTITY_TYPE.LOCATION) {
			String tokenText = "";
			if (type == ENTITY_TYPE.NUMBER || type == ENTITY_TYPE.ORDINAL || type == ENTITY_TYPE.DATE) {
				boolean isAddress = false;
				if (i + 4 < tokens.size()) {
					for (int x = 1; x <= 4; x++) {
						String nextne = tokens.get(i + x).get(NamedEntityTagAnnotation.class);
						if (nextne.equals(ENTITY_TYPE.LOCATION.toString()))
							isAddress = true;
					}
				}
				if (isAddress) {
					for (int x = 0; x < 4; x++) {
						tokenText += tokens.get(i + x).originalText() + " ";
					}
					i += 4;
					location.add(tokenText);
				}
			}
		}
		return i;
	}


	private static List<CoreMap> getEntities(StanfordCoreNLP stanfordNERPipeline, String text) {
		Annotation document = new Annotation(text);
		stanfordNERPipeline.annotate(document);

		return document.get(SentencesAnnotation.class);
	}


	private static boolean neededEntity(String ne) {
		return ne.equals(ENTITY_TYPE.PERSON.toString()) || ne.equals(ENTITY_TYPE.LOCATION.toString())
				|| ne.equals(ENTITY_TYPE.NUMBER.toString()) || ne.equals(ENTITY_TYPE.ORGANIZATION.toString())
				|| ne.equals(ENTITY_TYPE.ORDINAL.toString()) || ne.equals(ENTITY_TYPE.DATE.toString());
	}

	private static boolean isNullOrEmpty(String str) {
		return str == null || str.equals("") || str.equals("null") || str.equals("NULL");
	}
}
