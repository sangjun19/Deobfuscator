	.file	"manikanta-kondeti_linux-Terminal_listing4_flatten.c"
	.text
	.globl	_TIG_IZ_2wgJ_argc
	.bss
	.align 4
	.type	_TIG_IZ_2wgJ_argc, @object
	.size	_TIG_IZ_2wgJ_argc, 4
_TIG_IZ_2wgJ_argc:
	.zero	4
	.globl	_TIG_IZ_2wgJ_envp
	.align 8
	.type	_TIG_IZ_2wgJ_envp, @object
	.size	_TIG_IZ_2wgJ_envp, 8
_TIG_IZ_2wgJ_envp:
	.zero	8
	.globl	_TIG_IZ_2wgJ_argv
	.align 8
	.type	_TIG_IZ_2wgJ_argv, @object
	.size	_TIG_IZ_2wgJ_argv, 8
_TIG_IZ_2wgJ_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"/bin/pwd"
.LC1:
	.string	"%c\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_2wgJ_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_2wgJ_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_2wgJ_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 148 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-2wgJ--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_2wgJ_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_2wgJ_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_2wgJ_envp(%rip)
	nop
	movq	$4, -32(%rbp)
.L22:
	cmpq	$11, -32(%rbp)
	ja	.L25
	movq	-32(%rbp), %rax
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
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L25-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L25-.L8
	.long	.L25-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L25-.L8
	.long	.L7-.L8
	.text
.L12:
	movq	$0, -32(%rbp)
	jmp	.L16
.L10:
	movq	-72(%rbp), %rdx
	movq	-64(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	execve@PLT
	movq	$1, -32(%rbp)
	jmp	.L16
.L14:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L23
	jmp	.L24
.L7:
	movzbl	-37(%rbp), %eax
	movsbl	%al, %eax
	cmpl	$10, %eax
	jne	.L18
	movq	$5, -32(%rbp)
	jmp	.L19
.L18:
	movq	$2, -32(%rbp)
	nop
.L19:
	jmp	.L16
.L9:
	call	getchar@PLT
	movl	%eax, -36(%rbp)
	movl	-36(%rbp), %eax
	movb	%al, -37(%rbp)
	movzbl	-37(%rbp), %eax
	movsbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -32(%rbp)
	jmp	.L16
.L11:
	leaq	-18(%rbp), %rax
	movq	$0, (%rax)
	movw	$0, 8(%rax)
	movq	$0, -32(%rbp)
	jmp	.L16
.L15:
	movzbl	-37(%rbp), %eax
	cmpb	$10, %al
	je	.L20
	movq	$9, -32(%rbp)
	jmp	.L16
.L20:
	movq	$8, -32(%rbp)
	jmp	.L16
.L13:
	leaq	-37(%rbp), %rcx
	leaq	-18(%rbp), %rax
	movl	$1, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncat@PLT
	movq	$0, -32(%rbp)
	jmp	.L16
.L25:
	nop
.L16:
	jmp	.L22
.L24:
	call	__stack_chk_fail@PLT
.L23:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
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
