	.file	"Bobleeswagger09_alx-low_level_programming_100-main_opcodes_flatten.c"
	.text
	.globl	_TIG_IZ_BKJE_envp
	.bss
	.align 8
	.type	_TIG_IZ_BKJE_envp, @object
	.size	_TIG_IZ_BKJE_envp, 8
_TIG_IZ_BKJE_envp:
	.zero	8
	.globl	_TIG_IZ_BKJE_argv
	.align 8
	.type	_TIG_IZ_BKJE_argv, @object
	.size	_TIG_IZ_BKJE_argv, 8
_TIG_IZ_BKJE_argv:
	.zero	8
	.globl	_TIG_IZ_BKJE_argc
	.align 4
	.type	_TIG_IZ_BKJE_argc, @object
	.size	_TIG_IZ_BKJE_argc, 4
_TIG_IZ_BKJE_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Error"
.LC1:
	.string	"%02hhx "
.LC2:
	.string	"%02hhx\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	$0, _TIG_IZ_BKJE_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_BKJE_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_BKJE_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 102 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-BKJE--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_BKJE_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_BKJE_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_BKJE_envp(%rip)
	nop
	movq	$12, -8(%rbp)
.L29:
	cmpq	$17, -8(%rbp)
	ja	.L31
	movq	-8(%rbp), %rax
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
	.long	.L18-.L8
	.long	.L31-.L8
	.long	.L17-.L8
	.long	.L31-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L31-.L8
	.long	.L13-.L8
	.long	.L31-.L8
	.long	.L31-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L31-.L8
	.long	.L31-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L16:
	leaq	main(%rip), %rax
	movq	%rax, -16(%rbp)
	movl	$0, -20(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L19
.L10:
	cmpl	$0, -24(%rbp)
	jns	.L20
	movq	$8, -8(%rbp)
	jmp	.L19
.L20:
	movq	$4, -8(%rbp)
	jmp	.L19
.L11:
	cmpl	$2, -36(%rbp)
	je	.L22
	movq	$5, -8(%rbp)
	jmp	.L19
.L22:
	movq	$16, -8(%rbp)
	jmp	.L19
.L13:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$2, %edi
	call	exit@PLT
.L9:
	movq	-48(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -24(%rbp)
	movq	$15, -8(%rbp)
	jmp	.L19
.L12:
	movl	-20(%rbp), %eax
	movslq	%eax, %rdx
	movq	-16(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -20(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L19
.L7:
	movl	-20(%rbp), %eax
	cmpl	-24(%rbp), %eax
	jge	.L24
	movq	$6, -8(%rbp)
	jmp	.L19
.L24:
	movq	$2, -8(%rbp)
	jmp	.L19
.L14:
	movl	-24(%rbp), %eax
	subl	$1, %eax
	cmpl	%eax, -20(%rbp)
	jne	.L26
	movq	$0, -8(%rbp)
	jmp	.L19
.L26:
	movq	$11, -8(%rbp)
	jmp	.L19
.L15:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$1, %edi
	call	exit@PLT
.L18:
	movl	-20(%rbp), %eax
	movslq	%eax, %rdx
	movq	-16(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -8(%rbp)
	jmp	.L19
.L17:
	movl	$0, %eax
	jmp	.L30
.L31:
	nop
.L19:
	jmp	.L29
.L30:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
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
