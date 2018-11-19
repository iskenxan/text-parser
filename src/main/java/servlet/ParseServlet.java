package servlet;

import java.io.IOException;
import org.json.JSONObject;
import javax.servlet.ServletException;
import javax.servlet.ServletOutputStream;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import java.util.Properties;
import model.NdaParser;


@WebServlet(
        name = "ParseServlet",
        urlPatterns = {"/parse-file"},
        loadOnStartup = 1
    )
public class ParseServlet extends HttpServlet {
	public static StanfordCoreNLP stanfordNERPipeline;

  @Override
  	public void init() throws ServletException {
  		super.init();
      initializePipeline();
  	}


    private static void initializePipeline() {
      Properties props = new Properties();
      props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner");
      props.setProperty("coref.algorithm", "neural");
      props.setProperty("ner.useSUTime", "false");
      stanfordNERPipeline = new StanfordCoreNLP(props);
      System.out.println("done loading " + stanfordNERPipeline.toString());
    }


    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp)
            throws ServletException, IOException {
        ServletOutputStream out = resp.getOutputStream();
        out.write("hello heroku".getBytes());
        out.flush();
        out.close();
    }

    @Override
    protected void doPost(HttpServletRequest req, HttpServletResponse resp)
            throws ServletException, IOException {
        String text = req.getParameter("text");
        String name = req.getParameter("user_name");
        String company = req.getParameter("user_company");
        JSONObject result = null;
        try {
           result = NdaParser.parseForOtherPartyInfo(stanfordNERPipeline, text, name, company);
           ServletOutputStream out = resp.getOutputStream();
           out.write(result.toString().getBytes());
           out.flush();
           out.close();
        } catch(Exception e) {
          e.printStackTrace();
          ServletOutputStream out = resp.getOutputStream();
          out.write(e.getMessage().getBytes());
          out.flush();
          out.close();
        }
    }
}
