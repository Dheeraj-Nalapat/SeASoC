/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE259_Hard_Coded_Password__w32_char_31.c
Label Definition File: CWE259_Hard_Coded_Password__w32.label.xml
Template File: sources-sink-31.tmpl.c
*/
/*
 * @description
 * CWE: 259 Use of Hard-coded Password
 * BadSource:  Use a hardcoded password
 * GoodSource: Read the password from the console
 * Sinks:
 *    BadSink : Authenticate the user using LogonUserA()
 * Flow Variant: 31 Data flow using a copy of data within the same function
 *
 * */

#include "std_testcase.h"

#include <wchar.h>
#define PASSWORD_SZ 100

#define PASSWORD "ABCD1234!"

#ifdef _WIN32
#include <windows.h>
#pragma comment(lib, "advapi32.lib")
#endif

#ifndef OMITBAD

void CWE259_Hard_Coded_Password__w32_char_31_bad()
{
    char * password;
    char password_buf[100] = "";
    password = password_buf;
    /* FLAW: Use a hardcoded password */
    strcpy(password, PASSWORD);
    {
        char * password_copy = password;
        char * password = password_copy;
        {
#ifdef _WIN32 /* this is WIN32 specific */
            HANDLE pHandle;
            char * username = "User";
            char * domain = "Domain";
            /* POTENTIAL FLAW: Attempt to login user with password from the source (which may be hardcoded) */
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
#endif
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B()
{
    char * password;
    char password_buf[100] = "";
    password = password_buf;
    {
        size_t password_len = 0;
        /* FIX: Read the password from the console */
        fgets(password, 100, stdin);
        /* Remove the carriage return from the string that is inserted by fgets() */
        password_len = strlen(password);
        if (password_len > 0)
        {
            password[password_len-1] = '\0';
        }
    }
    {
        char * password_copy = password;
        char * password = password_copy;
        {
#ifdef _WIN32 /* this is WIN32 specific */
            HANDLE pHandle;
            char * username = "User";
            char * domain = "Domain";
            /* POTENTIAL FLAW: Attempt to login user with password from the source (which may be hardcoded) */
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
#endif
        }
    }
}

void CWE259_Hard_Coded_Password__w32_char_31_good()
{
    goodG2B();
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
    CWE259_Hard_Coded_Password__w32_char_31_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE259_Hard_Coded_Password__w32_char_31_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
