/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_05.c
Label Definition File: CWE226_Sensitive_Information_Uncleared_Before_Release.label.xml
Template File: point-flaw-05.tmpl.c
*/
/*
 * @description
 * CWE: 226 Sensitive Information Uncleared Before Release
 * Sinks: w32alloca
 *    GoodSink: Clear the password buffer before releasing the memory from the stack
 *    BadSink : Release password from the stack without first clearing the buffer
 * Flow Variant: 05 Control flow: if(static_t) and if(static_f)
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifdef _WIN32
# include <windows.h>
#pragma comment(lib, "advapi32.lib")
#endif

/* The two variables below are not defined as "const", but are never
   assigned any other value, so a tool should be able to identify that
   reads of these will always return their initialized values. */
static int static_t = 1; /* true */
static int static_f = 0; /* false */

#ifndef OMITBAD

void CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_05_bad()
{
    if(static_t)
    {
        {
            char * password = (char *)ALLOCA(100*sizeof(char));
            size_t password_len = 0;
            HANDLE pHandle;
            char * username = "User";
            char * domain = "Domain";
            /* Initialize password */
            password[0] = '\0';
            fgets(password, 100, stdin);
            /* Remove the carriage return from the string that is inserted by fgets() */
            password_len = strlen(password);
            if (password_len > 0)
            {
                password[password_len-1] = '\0';
            }
            /* Use the password in LogonUser() to establish that it is "sensitive" */
            if (LogonUserA(
                        username,
                        domain,
                        password,
                        LOGON32_LOGON_NETWORK,
                        LOGON32_PROVIDER_DEFAULT,
                        &pHandle) != 0)
            {
                printLine("User logge.d in successfully.");
                CloseHandle(pHandle);
            }
            else
            {
                printLine("Unable to login.");
            }
            /* FLAW: Release password from the stack without first clearing the buffer */
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            char * password = (char *)ALLOCA(100*sizeof(char));
            size_t password_len = 0;
            HANDLE pHandle;
            char * username = "User";
            char * domain = "Domain";
            /* Initialize password */
            password[0] = '\0';
            fgets(password, 100, stdin);
            /* Remove the carriage return from the string that is inserted by fgets() */
            password_len = strlen(password);
            if (password_len > 0)
            {
                password[password_len-1] = '\0';
            }
            /* Use the password in LogonUser() to establish that it is "sensitive" */
            if (LogonUserA(
                        username,
                        domain,
                        password,
                        LOGON32_LOGON_NETWORK,
                        LOGON32_PROVIDER_DEFAULT,
                        &pHandle) != 0)
            {
                printLine("User logged in successfully.");
                CloseHandle(pHandle);
            }
            else
            {
                printLine("Unable to login.");
            }
            password_len = strlen(password);
            /* FIX: Clear password prior to release from stack */
            SecureZeroMemory(password, password_len * sizeof(char));
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(static_f) instead of if(static_t) */
static void good1()
{
    if(static_f)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            char * password = (char *)ALLOCA(100*sizeof(char));
            size_t password_len = 0;
            HANDLE pHandle;
            char * username = "User";
            char * domain = "Domain";
            /* Initialize password */
            password[0] = '\0';
            fgets(password, 100, stdin);
            /* Remove the carriage return from the string that is inserted by fgets() */
            password_len = strlen(password);
            if (password_len > 0)
            {
                password[password_len-1] = '\0';
            }
            /* Use the password in LogonUser() to establish that it is "sensitive" */
            if (LogonUserA(
                        username,
                        domain,
                        password,
                        LOGON32_LOGON_NETWORK,
                        LOGON32_PROVIDER_DEFAULT,
                        &pHandle) != 0)
            {
                printLine("User logge.d in successfully.");
                CloseHandle(pHandle);
            }
            else
            {
                printLine("Unable to login.");
            }
            /* FLAW: Release password from the stack without first clearing the buffer */
        }
    }
    else
    {
        {
            char * password = (char *)ALLOCA(100*sizeof(char));
            size_t password_len = 0;
            HANDLE pHandle;
            char * username = "User";
            char * domain = "Domain";
            /* Initialize password */
            password[0] = '\0';
            fgets(password, 100, stdin);
            /* Remove the carriage return from the string that is inserted by fgets() */
            password_len = strlen(password);
            if (password_len > 0)
            {
                password[password_len-1] = '\0';
            }
            /* Use the password in LogonUser() to establish that it is "sensitive" */
            if (LogonUserA(
                        username,
                        domain,
                        password,
                        LOGON32_LOGON_NETWORK,
                        LOGON32_PROVIDER_DEFAULT,
                        &pHandle) != 0)
            {
                printLine("User logged in successfully.");
                CloseHandle(pHandle);
            }
            else
            {
                printLine("Unable to login.");
            }
            password_len = strlen(password);
            /* FIX: Clear password prior to release from stack */
            SecureZeroMemory(password, password_len * sizeof(char));
        }
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(static_t)
    {
        {
            char * password = (char *)ALLOCA(100*sizeof(char));
            size_t password_len = 0;
            HANDLE pHandle;
            char * username = "User";
            char * domain = "Domain";
            /* Initialize password */
            password[0] = '\0';
            fgets(password, 100, stdin);
            /* Remove the carriage return from the string that is inserted by fgets() */
            password_len = strlen(password);
            if (password_len > 0)
            {
                password[password_len-1] = '\0';
            }
            /* Use the password in LogonUser() to establish that it is "sensitive" */
            if (LogonUserA(
                        username,
                        domain,
                        password,
                        LOGON32_LOGON_NETWORK,
                        LOGON32_PROVIDER_DEFAULT,
                        &pHandle) != 0)
            {
                printLine("User logged in successfully.");
                CloseHandle(pHandle);
            }
            else
            {
                printLine("Unable to login.");
            }
            password_len = strlen(password);
            /* FIX: Clear password prior to release from stack */
            SecureZeroMemory(password, password_len * sizeof(char));
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            char * password = (char *)ALLOCA(100*sizeof(char));
            size_t password_len = 0;
            HANDLE pHandle;
            char * username = "User";
            char * domain = "Domain";
            /* Initialize password */
            password[0] = '\0';
            fgets(password, 100, stdin);
            /* Remove the carriage return from the string that is inserted by fgets() */
            password_len = strlen(password);
            if (password_len > 0)
            {
                password[password_len-1] = '\0';
            }
            /* Use the password in LogonUser() to establish that it is "sensitive" */
            if (LogonUserA(
                        username,
                        domain,
                        password,
                        LOGON32_LOGON_NETWORK,
                        LOGON32_PROVIDER_DEFAULT,
                        &pHandle) != 0)
            {
                printLine("User logge.d in successfully.");
                CloseHandle(pHandle);
            }
            else
            {
                printLine("Unable to login.");
            }
            /* FLAW: Release password from the stack without first clearing the buffer */
        }
    }
}

void CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_05_good()
{
    good1();
    good2();
}

#endif /* OMITGOOD */

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_05_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_05_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
