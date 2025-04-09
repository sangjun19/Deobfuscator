	.file	"shubhamguptaji_C-codes_exp57_flatten.c"
	.text
	.globl	_TIG_IZ_BT2j_envp
	.bss
	.align 8
	.type	_TIG_IZ_BT2j_envp, @object
	.size	_TIG_IZ_BT2j_envp, 8
_TIG_IZ_BT2j_envp:
	.zero	8
	.globl	_TIG_IZ_BT2j_argc
	.align 4
	.type	_TIG_IZ_BT2j_argc, @object
	.size	_TIG_IZ_BT2j_argc, 4
_TIG_IZ_BT2j_argc:
	.zero	4
	.globl	_TIG_IZ_BT2j_argv
	.align 8
	.type	_TIG_IZ_BT2j_argv, @object
	.size	_TIG_IZ_BT2j_argv, 8
_TIG_IZ_BT2j_argv:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"Error in opening first file !!"
.LC1:
	.string	"Enter name of first file name"
	.align 8
.LC2:
	.string	"Enter name of second file name"
.LC3:
	.string	"r"
.LC4:
	.string	"w"
.LC5:
	.string	"File copied!!!"
	.align 8
.LC6:
	.string	"Error in opening second file !!"
	.text
	.globl	main
	.type	main, @function
main:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$176, %rsp
	movl	%edi, -148(%rbp)
	movq	%rsi, -160(%rbp)
	movq	%rdx, -168(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_BT2j_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_BT2j_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_BT2j_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 129 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-BT2j--0
# 0 "" 2
#NO_APP
	movl	-148(%rbp), %eax
	movl	%eax, _TIG_IZ_BT2j_argc(%rip)
	movq	-160(%rbp), %rax
	movq	%rax, _TIG_IZ_BT2j_argv(%rip)
	movq	-168(%rbp), %rax
	movq	%rax, _TIG_IZ_BT2j_envp(%rip)
	nop
	movq	$10, -120(%rbp)
.L29:
	cmpq	$14, -120(%rbp)
	ja	.L32
	movq	-120(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L32-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L33-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L32-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L17:
	cmpq	$0, -144(%rbp)
	jne	.L21
	movq	$14, -120(%rbp)
	jmp	.L23
.L21:
	movq	$13, -120(%rbp)
	jmp	.L23
.L7:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -120(%rbp)
	jmp	.L23
.L10:
	movq	-136(%rbp), %rdx
	leaq	-48(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fputs@PLT
	movq	$3, -120(%rbp)
	jmp	.L23
.L13:
	cmpq	$0, -128(%rbp)
	je	.L24
	movq	$12, -120(%rbp)
	jmp	.L23
.L24:
	movq	$5, -120(%rbp)
	jmp	.L23
.L20:
	movq	-144(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-136(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$6, -120(%rbp)
	jmp	.L23
.L18:
	movq	-144(%rbp), %rdx
	leaq	-48(%rbp), %rax
	movl	$30, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	%rax, -128(%rbp)
	movq	$8, -120(%rbp)
	jmp	.L23
.L12:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	gets@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-80(%rbp), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	gets@PLT
	leaq	-112(%rbp), %rax
	leaq	.LC3(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -144(%rbp)
	movq	$4, -120(%rbp)
	jmp	.L23
.L9:
	leaq	-80(%rbp), %rax
	leaq	.LC4(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -136(%rbp)
	movq	$2, -120(%rbp)
	jmp	.L23
.L16:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -120(%rbp)
	jmp	.L23
.L11:
	movq	$9, -120(%rbp)
	jmp	.L23
.L14:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -120(%rbp)
	jmp	.L23
.L19:
	cmpq	$0, -136(%rbp)
	jne	.L27
	movq	$7, -120(%rbp)
	jmp	.L23
.L27:
	movq	$3, -120(%rbp)
	jmp	.L23
.L32:
	nop
.L23:
	jmp	.L29
.L33:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L31
	call	__stack_chk_fail@PLT
.L31:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	main, .-main
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
