/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE78_OS_Command_Injection__wchar_t_Environment_system_09.c
Label Definition File: CWE78_OS_Command_Injection.fullpath.label.xml
Template File: sources-sink-09.tmpl.c
*/
/*
 * @description
 * CWE: 78 OS Command Injection
 * BadSource: Environment Read input from an environment variable
 * GoodSource: Benign input
 * Sink: system
 *    BadSink : Execute command using system()
 * Flow Variant: 09 Control flow: if(global_const_t) and if(global_const_f)
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifdef _WIN32
# define COMMAND_INT_PATH L"%WINDIR%\\system32\\cmd.exe"
# define COMMAND_INT L"cmd.exe"
# define COMMAND_ARG1 L"/c"
# define COMMAND_ARG2 L"dir"
# define FULL_COMMAND COMMAND_INT_PATH L" " COMMAND_ARG1 L" "COMMAND_ARG2 L" "
#else /* NOT _WIN32 */
# define COMMAND_INT_PATH L"/bin/sh"
# define COMMAND_INT L"sh"
# define COMMAND_ARG1 L"ls"
# define FULL_COMMAND COMMAND_INT_PATH L" " COMMAND_ARG1 L" "
#endif

#define ENV_VARIABLE L"ADD"

#ifdef _WIN32
# define GETENV _wgetenv
#else
# define GETENV wgetenv
#endif

#ifndef OMITBAD

void CWE78_OS_Command_Injection__wchar_t_Environment_system_09_bad()
{
    wchar_t * data;
    wchar_t data_buf[100] = FULL_COMMAND;
    data = data_buf;
    if(global_const_t)
    {
        {
            /* Read input from an environment variable */
            size_t data_len = wcslen(data);
            wchar_t * environment = GETENV(ENV_VARIABLE);
            /* If there is data in the environment variable */
            if (environment != NULL)
            {
                wcsncat(data+data_len, environment, 100-data_len-1);
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Benign input preventing command injection */
        wcscat(data, L"*.*");
    }
    /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
    _wsystem(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B1() - use goodsource and badsink by changing the global_const_t to global_const_f */
static void goodG2B1()
{
    wchar_t * data;
    wchar_t data_buf[100] = FULL_COMMAND;
    data = data_buf;
    if(global_const_f)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* Read input from an environment variable */
            size_t data_len = wcslen(data);
            wchar_t * environment = GETENV(ENV_VARIABLE);
            /* If there is data in the environment variable */
            if (environment != NULL)
            {
                wcsncat(data+data_len, environment, 100-data_len-1);
            }
        }
    }
    else
    {
        /* FIX: Benign input preventing command injection */
        wcscat(data, L"*.*");
    }
    /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
    _wsystem(data);
}

/* goodG2B2() - use goodsource and badsink by reversing the blocks in the if statement */
static void goodG2B2()
{
    wchar_t * data;
    wchar_t data_buf[100] = FULL_COMMAND;
    data = data_buf;
    if(global_const_t)
    {
        /* FIX: Benign input preventing command injection */
        wcscat(data, L"*.*");
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* Read input from an environment variable */
            size_t data_len = wcslen(data);
            wchar_t * environment = GETENV(ENV_VARIABLE);
            /* If there is data in the environment variable */
            if (environment != NULL)
            {
                wcsncat(data+data_len, environment, 100-data_len-1);
            }
        }
    }
    /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
    _wsystem(data);
}

void CWE78_OS_Command_Injection__wchar_t_Environment_system_09_good()
{
    goodG2B1();
    goodG2B2();
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
    CWE78_OS_Command_Injection__wchar_t_Environment_system_09_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE78_OS_Command_Injection__wchar_t_Environment_system_09_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
