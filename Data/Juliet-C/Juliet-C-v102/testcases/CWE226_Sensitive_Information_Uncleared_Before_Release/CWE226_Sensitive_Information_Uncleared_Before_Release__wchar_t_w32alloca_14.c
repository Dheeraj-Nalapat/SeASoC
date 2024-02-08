/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_14.c
Label Definition File: CWE226_Sensitive_Information_Uncleared_Before_Release.label.xml
Template File: point-flaw-14.tmpl.c
*/
/*
 * @description
 * CWE: 226 Sensitive Information Uncleared Before Release
 * Sinks: w32alloca
 *    GoodSink: Clear the password buffer before releasing the memory from the stack
 *    BadSink : Release password from the stack without first clearing the buffer
 * Flow Variant: 14 Control flow: if(global_five==5) and if(global_five!=5)
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifdef _WIN32
# include <windows.h>
#pragma comment(lib, "advapi32.lib")
#endif

#ifndef OMITBAD

void CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_14_bad()
{
    if(global_five==5)
    {
        {
            wchar_t * password = (wchar_t *)ALLOCA(100*sizeof(wchar_t));
            size_t password_len = 0;
            HANDLE pHandle;
            wchar_t * username = L"User";
            wchar_t * domain = L"Domain";
            /* Initialize password */
            password[0] = L'\0';
            fgetws(password, 100, stdin);
            /* Remove the carriage return from the string that is inserted by fgetws() */
            password_len = wcslen(password);
            if (password_len > 0)
            {
                password[password_len-1] = L'\0';
            }
            /* Use the password in LogonUser() to establish that it is "sensitive" */
            if (LogonUserW(
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
            wchar_t * password = (wchar_t *)ALLOCA(100*sizeof(wchar_t));
            size_t password_len = 0;
            HANDLE pHandle;
            wchar_t * username = L"User";
            wchar_t * domain = L"Domain";
            /* Initialize password */
            password[0] = L'\0';
            fgetws(password, 100, stdin);
            /* Remove the carriage return from the string that is inserted by fgetws() */
            password_len = wcslen(password);
            if (password_len > 0)
            {
                password[password_len-1] = L'\0';
            }
            /* Use the password in LogonUser() to establish that it is "sensitive" */
            if (LogonUserW(
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
            password_len = wcslen(password);
            /* FIX: Clear password prior to release from stack */
            SecureZeroMemory(password, password_len * sizeof(wchar_t));
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(global_five!=5) instead of if(global_five==5) */
static void good1()
{
    if(global_five!=5)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            wchar_t * password = (wchar_t *)ALLOCA(100*sizeof(wchar_t));
            size_t password_len = 0;
            HANDLE pHandle;
            wchar_t * username = L"User";
            wchar_t * domain = L"Domain";
            /* Initialize password */
            password[0] = L'\0';
            fgetws(password, 100, stdin);
            /* Remove the carriage return from the string that is inserted by fgetws() */
            password_len = wcslen(password);
            if (password_len > 0)
            {
                password[password_len-1] = L'\0';
            }
            /* Use the password in LogonUser() to establish that it is "sensitive" */
            if (LogonUserW(
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
            wchar_t * password = (wchar_t *)ALLOCA(100*sizeof(wchar_t));
            size_t password_len = 0;
            HANDLE pHandle;
            wchar_t * username = L"User";
            wchar_t * domain = L"Domain";
            /* Initialize password */
            password[0] = L'\0';
            fgetws(password, 100, stdin);
            /* Remove the carriage return from the string that is inserted by fgetws() */
            password_len = wcslen(password);
            if (password_len > 0)
            {
                password[password_len-1] = L'\0';
            }
            /* Use the password in LogonUser() to establish that it is "sensitive" */
            if (LogonUserW(
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
            password_len = wcslen(password);
            /* FIX: Clear password prior to release from stack */
            SecureZeroMemory(password, password_len * sizeof(wchar_t));
        }
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(global_five==5)
    {
        {
            wchar_t * password = (wchar_t *)ALLOCA(100*sizeof(wchar_t));
            size_t password_len = 0;
            HANDLE pHandle;
            wchar_t * username = L"User";
            wchar_t * domain = L"Domain";
            /* Initialize password */
            password[0] = L'\0';
            fgetws(password, 100, stdin);
            /* Remove the carriage return from the string that is inserted by fgetws() */
            password_len = wcslen(password);
            if (password_len > 0)
            {
                password[password_len-1] = L'\0';
            }
            /* Use the password in LogonUser() to establish that it is "sensitive" */
            if (LogonUserW(
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
            password_len = wcslen(password);
            /* FIX: Clear password prior to release from stack */
            SecureZeroMemory(password, password_len * sizeof(wchar_t));
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            wchar_t * password = (wchar_t *)ALLOCA(100*sizeof(wchar_t));
            size_t password_len = 0;
            HANDLE pHandle;
            wchar_t * username = L"User";
            wchar_t * domain = L"Domain";
            /* Initialize password */
            password[0] = L'\0';
            fgetws(password, 100, stdin);
            /* Remove the carriage return from the string that is inserted by fgetws() */
            password_len = wcslen(password);
            if (password_len > 0)
            {
                password[password_len-1] = L'\0';
            }
            /* Use the password in LogonUser() to establish that it is "sensitive" */
            if (LogonUserW(
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

void CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_14_good()
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
    CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_14_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_14_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
