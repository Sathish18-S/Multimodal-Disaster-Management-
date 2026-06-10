import * as React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';

// Import screens
import Home from './screens/Home';
import Tweet from './screens/Tweet';

const Stack = createStackNavigator();

export default function App() {
    return ( <NavigationContainer>
  <Stack.Navigator initialRouteName="Home">
    <Stack.Screen 
      name="Home"
      component={Home}
      options={{ headerShown: false }}
    />
    <Stack.Screen 
      name="Tweet"
      component={Tweet}
      options={{ headerShown: false }}  // This removes the header completely
    />
  </Stack.Navigator>
</NavigationContainer>
    );
}