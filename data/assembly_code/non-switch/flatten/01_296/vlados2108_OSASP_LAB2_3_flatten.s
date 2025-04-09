	.file	"vlados2108_OSASP_LAB2_3_flatten.c"
	.text
	.globl	_TIG_IZ_StD5_argv
	.bss
	.align 8
	.type	_TIG_IZ_StD5_argv, @object
	.size	_TIG_IZ_StD5_argv, 8
_TIG_IZ_StD5_argv:
	.zero	8
	.globl	_TIG_IZ_StD5_envp
	.align 8
	.type	_TIG_IZ_StD5_envp, @object
	.size	_TIG_IZ_StD5_envp, 8
_TIG_IZ_StD5_envp:
	.zero	8
	.globl	_TIG_IZ_StD5_argc
	.align 4
	.type	_TIG_IZ_StD5_argc, @object
	.size	_TIG_IZ_StD5_argc, 4
_TIG_IZ_StD5_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Can't close file\n"
	.align 8
.LC1:
	.string	"Inalid number of params\nThere must be one parametr - name of File"
	.align 8
.LC2:
	.string	"Inalid parametr\nThere must be one parametr - name of File"
.LC3:
	.string	"Can't read symbol\n"
.LC4:
	.string	"w"
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
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
	movq	$0, _TIG_IZ_StD5_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_StD5_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_StD5_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 106 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-StD5--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_StD5_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_StD5_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_StD5_envp(%rip)
	nop
	movq	$0, -8(%rbp)
.L41:
	cmpq	$20, -8(%rbp)
	ja	.L42
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
	.long	.L26-.L8
	.long	.L25-.L8
	.long	.L42-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L42-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L10:
	movl	$0, %eax
	jmp	.L27
.L23:
	movl	$-1, %eax
	jmp	.L27
.L14:
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fgetc@PLT
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movb	%al, -25(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L28
.L13:
	movl	$0, %eax
	jmp	.L27
.L16:
	movsbl	-25(%rbp), %eax
	movq	-16(%rbp), %rdx
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	fputc@PLT
	movq	$5, -8(%rbp)
	jmp	.L28
.L20:
	movl	$-1, %eax
	jmp	.L27
.L25:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$17, %edx
	movl	$1, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$8, -8(%rbp)
	jmp	.L28
.L24:
	cmpb	$42, -25(%rbp)
	je	.L29
	movq	$12, -8(%rbp)
	jmp	.L28
.L29:
	movq	$5, -8(%rbp)
	jmp	.L28
.L12:
	cmpb	$-1, -25(%rbp)
	jne	.L31
	movq	$17, -8(%rbp)
	jmp	.L28
.L31:
	movq	$3, -8(%rbp)
	jmp	.L28
.L17:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$65, %edx
	movl	$1, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$15, -8(%rbp)
	jmp	.L28
.L19:
	movq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movl	%eax, -24(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L28
.L15:
	cmpl	$0, -24(%rbp)
	je	.L33
	movq	$1, -8(%rbp)
	jmp	.L28
.L33:
	movq	$18, -8(%rbp)
	jmp	.L28
.L9:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$57, %edx
	movl	$1, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$20, -8(%rbp)
	jmp	.L28
.L11:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$18, %edx
	movl	$1, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$4, -8(%rbp)
	jmp	.L28
.L21:
	movq	-48(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	.LC4(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -16(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L28
.L22:
	cmpb	$42, -25(%rbp)
	je	.L35
	movq	$14, -8(%rbp)
	jmp	.L28
.L35:
	movq	$9, -8(%rbp)
	jmp	.L28
.L18:
	cmpq	$0, -16(%rbp)
	je	.L37
	movq	$5, -8(%rbp)
	jmp	.L28
.L37:
	movq	$19, -8(%rbp)
	jmp	.L28
.L26:
	cmpl	$2, -36(%rbp)
	jne	.L39
	movq	$6, -8(%rbp)
	jmp	.L28
.L39:
	movq	$11, -8(%rbp)
	jmp	.L28
.L7:
	movl	$-1, %eax
	jmp	.L27
.L42:
	nop
.L28:
	jmp	.L41
.L27:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
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
