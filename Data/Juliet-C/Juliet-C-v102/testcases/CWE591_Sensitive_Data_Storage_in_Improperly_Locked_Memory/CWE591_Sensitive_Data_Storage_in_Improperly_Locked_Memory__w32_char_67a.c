/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE591_Sensitive_Data_Storage_in_Improperly_Locked_Memory__w32_char_67a.c
Label Definition File: CWE591_Sensitive_Data_Storage_in_Improperly_Locked_Memory__w32.label.xml
Template File: sources-sink-67a.tmpl.c
*/
/*
 * @description
 * CWE: 591 Sensitive Data Storage in Improperly Locked Memory
 * BadSource:  Allocate memory for sensitive data and use VirtualLock() to lock the buffer into memory
 * GoodSource: Allocate memory for sensitive data and use VirtualLock() to lock the buffer into memory
 * Sinks:
 *    BadSink : Authenticate the user using LogonUserA()
 * Flow Variant: 67 Data flow: data passed in a struct from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>
#ifdef _WIN32
#include <windows.h>
#endif

#ifdef _WIN32
#pragma comment(lib, "advapi32.lib")
#endif

typedef struct _CWE591_Sensitive_Data_Storage_in_Improperly_Locked_Memory__w32_char_67_struct_type
{
    char * a;
} CWE591_Sensitive_Data_Storage_in_Improperly_Locked_Memory__w32_char_67_struct_type;

#ifndef OMITBAD

/* bad function declaration */
void CWE591_Sensitive_Data_Storage_in_Improperly_Locked_Memory__w32_char_67b_bad_sink(CWE591_Sensitive_Data_Storage_in_Improperly_Locked_Memory__w32_char_67_struct_type my_struct);

void CWE591_Sensitive_Data_Storage_in_Improperly_Locked_Memory__w32_char_67_bad()
{
    char * password;
    CWE591_Sensitive_Data_Storage_in_Improperly_Locked_Memory__w32_char_67_struct_type my_struct;
    /* Initialize Data */
    password = "";
    password = (char *)malloc(100*sizeof(char));
    /* FLAW: Do not lock the memory */
    /* INCIDENTAL FLAW: CWE-259 Hardcoded Password */
    strcpy(password, "Password1234!");
    my_struct.a = password;
    CWE591_Sensitive_Data_Storage_in_Improperly_Locked_Memory__w32_char_67b_bad_sink(my_struct);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE591_Sensitive_Data_Storage_in_Improperly_Locked_Memory__w32_char_67b_goodG2B_sink(CWE591_Sensitive_Data_Storage_in_Improperly_Locked_Memory__w32_char_67_struct_type my_struct);

static void goodG2B()
{
    char * password;
    CWE591_Sensitive_Data_Storage_in_Improperly_Locked_Memory__w32_char_67_struct_type my_struct;
    /* Initialize Data */
    password = "";
#ifdef _WIN32 /* this is WIN32 specific */
    password = (char *)malloc(100*sizeof(char));
    /* FIX: Use VirtualLock() to lock the buffer into memory */
    if(!VirtualLock(password, 100*sizeof(char)))
    {
        printLine("Memory could not be locked");
        exit(1);
    }
    /* INCIDENTAL FLAW: CWE-259 Hardcoded Password */
    strcpy(password, "Password1234!");
#endif
    my_struct.a = password;
    CWE591_Sensitive_Data_Storage_in_Improperly_Locked_Memory__w32_char_67b_goodG2B_sink(my_struct);
}

void CWE591_Sensitive_Data_Storage_in_Improperly_Locked_Memory__w32_char_67_good()
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
    CWE591_Sensitive_Data_Storage_in_Improperly_Locked_Memory__w32_char_67_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE591_Sensitive_Data_Storage_in_Improperly_Locked_Memory__w32_char_67_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
