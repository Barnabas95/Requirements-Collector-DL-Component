package ch.zhaw.hassebjo;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class AppConfig {

    @Bean
    ClassifyUserStories classifyUserStories()
    {
        return new ClassifyUserStoriesImpl();
    }
}
