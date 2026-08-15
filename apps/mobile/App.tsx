import React from 'react';
import { SafeAreaView, StyleSheet, Platform } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { UNetStudioUniversal } from '@opencv-studio/shared';

const BACKEND_URL = Platform.OS === 'web' 
  ? 'http://localhost:8000' 
  : 'http://192.168.29.151:8000';

export default function App() {
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="light" />
      <UNetStudioUniversal baseUrl={BACKEND_URL} />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#060911',
  },
});
